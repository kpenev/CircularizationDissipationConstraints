"""Find all expected ML data entries from a given log file."""

import re
from glob import glob
from itertools import count


class SkipParams(BaseException):
    """Raised if parameter set was interrupted."""

    def __init__(self, index, match):
        self.index = index
        self.match = match


def find(rex, except_if_rex, log_file):
    """Return the first match of rex in the given log file (None if EOF)."""

    # print('Looking for ' + repr(rex))
    line = log_file.readline()
    while line:
        match = rex.match(line)
        if match:
            return match
        for bad_ind, bad_rex in enumerate(except_if_rex):
            except_match = bad_rex.match(line)
            if except_match:
                raise SkipParams(bad_ind, except_match)
        line = log_file.readline()

    return None


def get_line_number(line, fname):
    """Return the line number where the given line is in the given file."""

    with open(fname, "r", encoding="ascii") as file:
        for lineno in count():
            file_line = file.readline()
            if not file_line:
                return None
            if file_line.strip() == line.strip():
                return lineno
    return None


def get_line(lineno, fname):
    """Return the contents of the lineno line number of given file."""

    with open(fname, "r", encoding="ascii") as file:
        for _ in range(lineno):
            file.readline()
        return file.readline()


def parse(log_fname, data_fname, label_fname):
    """Parse the expected entries in ML training data from given log file."""

    float_re = "(?P<value>[e0-9.+-]*)"
    sample_rex = re.compile(r".*]\)\) -> Parameters:")
    params_1d = [
        "lgQ_min",
        "lgQ_break_period",
        "lgQ_powerlaw",
        "age",
        "feh",
        "final_orbital_period",
        "primary_mass",
        "secondary_mass",
        "cmd_primary_radius",
        "cmd_secondary_radius",
    ]
    param_rex = re.compile(f'^\t(?P<name>{"|".join(params_1d)}): {float_re}')
    initial_porb_rex = re.compile(
        "^DEBUG .* general_purpose_python_modules.solve_for_initial_values: "
        f"Initial period: {float_re}"
    )
    final_porb_rex = re.compile(
        "^DEBUG .* general_purpose_python_modules.solve_for_initial_values: "
        f"Final period: {float_re}"
    )
    max_tested_line = 0
    num_tested = 0
    with open(log_fname, "r", encoding="ascii") as log_file:
        while find(sample_rex, (), log_file):
            try:
                params = {}
                for _ in range(len(params_1d) - 1):
                    param_match = find(param_rex, (sample_rex,), log_file)
                    params[param_match["name"]] = param_match["value"]
                porb_initial_match = find(
                    initial_porb_rex, (sample_rex,), log_file
                )
                while porb_initial_match:
                    porb_initial = porb_initial_match["value"]
                    try:
                        final_porb_match = find(
                            final_porb_rex,
                            (initial_porb_rex, sample_rex),
                            log_file,
                        )
                        if final_porb_match is None:
                            break
                        else:
                            params["final_orbital_period"] = final_porb_match["value"]
                    except SkipParams as exc:
                        if exc.index == 0:
                            porb_initial_match = exc.match
                            continue
                        else:
                            raise

                    param_line = ",".join(params[name] for name in params_1d)
                    test_line_number = get_line_number(param_line, data_fname)
                    if test_line_number is None:
                        continue
                    if (
                        get_line(test_line_number, label_fname).strip()
                        == porb_initial.strip()
                    ):
                        max_tested_line = max(max_tested_line, test_line_number)
                        num_tested += 1
                    else:
                        print(
                            f"Mismatch on line {test_line_number}: "
                            f"{get_line(test_line_number, label_fname)!r} != "
                            f"{porb_initial!r} for {params!r}"
                        )
                    porb_initial_match = find(
                        initial_porb_rex, (sample_rex,), log_file
                    )
            except SkipParams:
                pass
    print(f"Tested {num_tested} entries from {log_fname!r}")
    print(f"Last tested ML line: {max_tested_line}")


if __name__ == "__main__":
    for log_fname in glob("ml_logs/logs/*.log"):
        print(f"Testing {log_fname}")
        parse(
            log_fname,
            "ml_logs/1dp/data.csv",
            "ml_logs/1dp/label.csv",
        )
