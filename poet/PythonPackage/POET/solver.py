"""Compatibility shim for legacy ``from POET.solver import poet_solver``.

The bundled POET interface used here provides direct orbital evolution, but
not the older ML initial-condition solver package. Sampling jobs should pass
``--disable-warm-start`` so this shim is not used at runtime.
"""


class _UnavailablePOETICSolver:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "POET_IC_Solver is unavailable in the bundled POET package; "
            "run with --disable-warm-start."
        )


class _PoetSolverModule:
    POET_IC_Solver = _UnavailablePOETICSolver


poet_solver = _PoetSolverModule()
