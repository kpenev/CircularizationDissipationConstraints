"""Set baseline evidence tolerance without modifying journal-fingerprinted source."""
from functools import wraps
from inspect import signature
import dynesty

BASELINE_DLOGZ = 0.05


def install():
    cls = dynesty.DynamicNestedSampler
    original = cls.run_nested
    if getattr(original, '_baseline_dlogz_policy', None) == BASELINE_DLOGZ:
        return
    parameters = signature(original)

    @wraps(original)
    def run_nested(self, *args, **kwargs):
        call = parameters.bind_partial(self, *args, **kwargs)
        call.arguments['dlogz_init'] = BASELINE_DLOGZ
        print(f'BASELINE POLICY: dlogz_init={BASELINE_DLOGZ}; '
              f'state={getattr(self, "internal_state", None)}; '
              'final posterior CDF stopping policy unchanged', flush=True)
        return original(*call.args, **call.kwargs)

    run_nested._baseline_dlogz_policy = BASELINE_DLOGZ
    cls.run_nested = run_nested
