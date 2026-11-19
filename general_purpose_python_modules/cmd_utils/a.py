def __init__(self, df, interp_var, section_keys=None):
    # section_keys are the columns that define a “section”
    section_keys = section_keys or []  # e.g., ["MH", "Age"] or ["FeH", "logAge"]
    self.interp_var = interp_var
    self.header = []
    self.isochrone_fname = "<dataframe>"

    if section_keys:
        self._data = []
        for _, g in df.sort_values(section_keys + [interp_var], kind="mergesort").groupby(section_keys, dropna=False):
            self._data.append(g.to_records(index=False))
    else:
        # fall back to single-section behavior
        self._data = [df.sort_values(interp_var, kind="mergesort").to_records(index=False)]

    self._validate_input_data()
    self._grid = tuple(
        (quantity, numpy.array(values))
        for quantity, values in self._get_interpolation_grid()
        if len(values) > 1
    )