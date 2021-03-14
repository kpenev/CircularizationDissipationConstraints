"""Functions used by multiple bayesian sampling scripts."""

import sys
import os
import os.path
from datetime import datetime
import logging

import git

def get_code_version_str():
    """Return a string identifying the version of the code being used."""

    repository = git.Repo(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)
                )
            )
        )
    )
    head_sha = repository.commit().hexsha
    if repository.is_dirty():
        return head_sha + ':dirty'
    return head_sha

def setup_process(config):
    """Logging and I/O setup for the current processes."""

    def ensure_directory(fname):
        """Make sure the directory containing the given name exists."""

        dirname = os.path.dirname(fname)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    fname_substitutions = dict(
        now=datetime.now().strftime(config.fname_datetime_format),
        system=config.system,
        pid=os.getpid()
    )

    std_out_err_fname = config.std_out_err_fname % fname_substitutions
    ensure_directory(std_out_err_fname)
    sys.stderr = open(std_out_err_fname, 'w', buffering=1)

    logging_fname = config.logging_fname % fname_substitutions
    ensure_directory(logging_fname)
    logging_config = dict(
        filename=logging_fname,
        level=getattr(logging, config.logging_verbosity.upper()),
        format=config.logging_message_format
    )
    if config.logging_datetime_format is not None:
        logging_config['datefmt'] = config.logging_datetime_format
    logging.basicConfig(**logging_config)
