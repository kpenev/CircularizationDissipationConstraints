# Publication validation

- All included runtime source hashes match the revised deployed Gaia snapshot.
- One integration-test fixture was updated for the current seven-system selection and retained covariance target; runtime code was unchanged.
- `python control/test_massive_policy.py`: 7 tests passed using mocked stellar quantities.
- Updated Gaia adapter integration tests: 6 tests passed in the actual Juno Python environment against the unchanged deployed modules and separately supplied observation inputs. No input tables are added to this branch.
- Envelope checks at 1, 7, 10.093766032005036 and 50 days passed.
- Generic Slurm scripts passed Bash syntax checks.
- The staged diff adds no research inputs, target lists, run records, compiled binaries or credentials.
- Newly authored documentation/examples passed whitespace checks. Historical deployed-source whitespace is preserved for hash fidelity.

An incomplete historical scratch test is excluded from publication and recorded in PYTHON_SYNTAX_AUDIT.json. The entire legacy tree is not claimed to be a runnable test suite. Native orbital convergence was not retested for publication; the ongoing runs were not modified.
