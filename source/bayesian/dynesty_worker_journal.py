"""Durable, ordered replay of completed Dynesty proposal worker tasks.

Only SamplerArgument calls are journaled. No live-point or posterior state is
changed here. Replay starts from the normal completed-iteration checkpoint.
An interrupted worker itself must restart; completed workers are reused.
"""

import fcntl
from contextlib import contextmanager
import hashlib
import logging
import os
import pickle
import shutil
from pathlib import Path
import tempfile

import numpy as np
import dynesty

_logger = logging.getLogger(__name__)
JOURNAL_VERSION = 1


def _atomic_pickle(path, value):
    """Commit one complete record, never expose a partially written pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            pickle.dump(value, stream, protocol=5)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _canonical(value):
    """Represent task inputs and random state without process-local pointers."""
    if isinstance(value, np.ndarray):
        return ('array', value.dtype.str, value.shape,
                np.ascontiguousarray(value).tobytes())
    if isinstance(value, np.random.SeedSequence):
        return ('SeedSequence', _canonical(value.state))
    if isinstance(value, np.random.Generator):
        return ('Generator', _canonical(value.bit_generator.state))
    if isinstance(value, dict):
        return ('dict', tuple((str(k), _canonical(v))
                              for k, v in sorted(value.items(),
                                                  key=lambda item: str(item[0]))))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_canonical(v) for v in value))
    if isinstance(value, np.generic):
        return ('numpy_scalar', value.dtype.str, value.tobytes())
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return value
    raise TypeError('Unsupported worker journal input: ' + str(type(value)))


def _inputs(argument):
    # Runtime likelihood/prior objects are rebuilt on restart. Their code and
    # configuration are bound by the namespace hash, not serialized native
    # interpolator pointers. These are all mathematical SamplerArgument inputs.
    return {name: getattr(argument, name) for name in
            ('u', 'loglstar', 'axes', 'scale', 'rseed', 'kwargs')}


def _signature(function, arguments):
    descriptor = (function.__module__, function.__qualname__,
                  tuple(_canonical(_inputs(arg)) for arg in arguments))
    return hashlib.sha256(pickle.dumps(descriptor, protocol=5)).hexdigest()


def _run_and_record(task):
    """Execute in a pool worker and persist before notifying the parent."""
    function, argument, result_path = task
    result = function(argument)
    _atomic_pickle(result_path, result)
    return result


def source_fingerprint(source_root):
    """Prevent reuse across changes to model/config or Dynesty implementation."""
    digest = hashlib.sha256(dynesty.__version__.encode())
    root = Path(source_root)
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix in ('.py', '.cfg'):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


class JournalPool:
    """Pool adapter recording each proposal group since the last checkpoint.

    A checkpoint content hash plus call sequence identifies each dispatch.
    On restore, the sampler regenerates the same arguments from saved RNG state.
    Completed task results are returned in their original order; only missing
    tasks execute again. Any signature mismatch fails closed.
    """

    def __init__(self, pool, checkpoint_file, source_root):
        self.pool = pool
        self.size = getattr(pool, '_processes', 1)
        self.checkpoint = Path(checkpoint_file)
        self.root = Path(checkpoint_file + '.worker_journal')
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = open(self.root / 'writer.lock', 'a+b')
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError('Another run already owns worker journal ' + str(self.root))
        self.namespace = source_fingerprint(source_root)
        self.generation = None
        self.checkpoint_stat = None
        self.sequence = 0
        self.previous_generation = None
        _logger.info('Worker journaling enabled: %s; source=%s',
                     self.root, self.namespace)

    def _generation(self):
        if not self.checkpoint.exists():
            return None
        stat = self.checkpoint.stat()
        key = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if key != self.checkpoint_stat:
            generation = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
            if generation != self.generation:
                self.previous_generation = self.generation
                self.generation = generation
                self.sequence = 0
                # Completed work is now inside the ordinary checkpoint. Keep
                # current and previous journals, discard older committed work.
                namespace_dir = self.root / self.namespace
                if namespace_dir.exists():
                    for directory in namespace_dir.iterdir():
                        if (directory.is_dir() and directory.name not in
                                (self.generation, self.previous_generation)):
                            shutil.rmtree(directory)
            self.checkpoint_stat = key
        return self.generation

    def map(self, function, iterable):
        arguments = list(iterable)
        # Bound updates, transforms, likelihood initialization and diagnostic
        # realizations retain their original map implementation.
        proposal = (arguments and all(
            type(arg).__name__ == 'SamplerArgument' and
            set(getattr(arg, '_fields', ())) ==
            {'u', 'loglstar', 'axes', 'scale', 'prior_transform',
             'loglikelihood', 'rseed', 'kwargs'} for arg in arguments))
        generation = self._generation() if proposal else None
        if generation is None:
            return self.pool.map(function, arguments)
        directory = self.root / self.namespace / generation / f'{self.sequence:08d}'
        self.sequence += 1
        manifest_path = directory / 'inputs.pkl'
        signature = _signature(function, arguments)
        if manifest_path.exists():
            with open(manifest_path, 'rb') as stream:
                manifest = pickle.load(stream)
            if (manifest['signature'] != signature or
                    manifest['version'] != JOURNAL_VERSION):
                raise RuntimeError('Worker journal replay mismatch at ' + str(directory))
        else:
            _atomic_pickle(manifest_path, dict(
                version=JOURNAL_VERSION, signature=signature,
                function=(function.__module__, function.__qualname__),
                inputs=[_inputs(arg) for arg in arguments]))
        results = [None] * len(arguments)
        missing = []
        for index, argument in enumerate(arguments):
            result_path = directory / f'result_{index:04d}.pkl'
            if result_path.exists():
                with open(result_path, 'rb') as stream:
                    results[index] = pickle.load(stream)
            else:
                missing.append((index, (function, argument, str(result_path))))
        _logger.info('Worker journal dispatch: checkpoint=%s group=%d '
                     'reused=%d pending=%d', generation[:12], self.sequence - 1,
                     len(arguments) - len(missing), len(missing))
        if missing:
            completed = self.pool.map(_run_and_record, [task for _, task in missing])
            for (index, _), result in zip(missing, completed):
                results[index] = result
        return results

    def close_journal(self):
        self.lock.close()


@contextmanager
def worker_journal_context(pool, checkpoint_file, source_root):
    """Keep the writer lock for the lifetime of this sampling process."""
    if pool is None or os.environ.get('NESTY_WORKER_JOURNAL', '1') != '1':
        yield pool
        return
    wrapper = JournalPool(pool, checkpoint_file, source_root)
    try:
        yield wrapper
    finally:
        wrapper.close_journal()
