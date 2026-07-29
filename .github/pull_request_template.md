## What and why

<!-- What changes, and what problem it solves. -->

## Checklist

- [ ] `pytest` passes
- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] If this touches `control.py`, `kalman.py`, or `framing.py`:
      `tests/test_closed_loop.py` still passes, or the assertions were changed
      deliberately and the reason is explained above
- [ ] New config keys are documented with a `#:` comment and have units in the name
- [ ] No new work added to the control loop without measuring its cost

## Tested on

<!-- Hardware, OS, and how. "Simulation only" is a fine answer — just say so. -->
