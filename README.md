# Lab 08 — The learner that remembers, and the repository that proves it

Content: Reinforcement Learning · Narxoz University

In Lab 07 your learner lived inside a notebook: it learned, printed a table, and forgot everything when the
session ended. In this lab you turn it into **a program somebody else can run**. It stops, **remembers what it
learned**, continues in a new session exactly where it was, and lives in a repository that shows it works.

There is no new algorithm. What is new is the engineering, and the repository is marked together with the
behaviour.

**Due: Thursday 10 December 2026, 23:59 (Asia/Almaty).** 15 points.

**No API key. No GPU. No cost.** There is no notebook in this lab: you need Python and git on your own computer.
**Use Python 3.12**: your program is checked with it, so your pinned versions must install on it. The
instructor's reference solution was run with Python 3.12 on an Apple-silicon Mac.

## Setup

1. Press **Use this template** → *Create a new repository*, choose **Public**, and give it a name such as
   `lab-08-AI-course`. Do **not** fork this repository.
   ([GitHub: creating a repository from a template](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template))
2. Clone your new repository.
3. **Commit as you work.** The history is part of what is marked; one large commit at the end shows no history.

## What you bring

Your three functions from Lab 07, with the same signatures:

```python
bellman_backup(p_next, reward, mask, v, gamma) -> np.ndarray
epsilon_greedy(q, state, epsilon, mask, rng) -> int
q_update(q, state, action, reward, next_state, alpha, gamma, mask) -> None
```

They go into the module **`shoplearn/learners.py`**, importable as `shoplearn.learners`. Test them with
`check_bellman_backup` and `check_q_learning` from `lab08_core` before you build anything on them.

## What is supplied

`lab08_core.py` is the Lab 07 file with **one addition**: `InventoryEnv.get_state()` and
`InventoryEnv.set_state()`, which return and restore everything the shop needs to continue (a round trip through
JSON is fine). Keep the file unchanged; you may move it into your package.

## The program

A package `shoplearn`, run as `python -m shoplearn`. The discount factor is γ = 0.95 everywhere.

### train

```
python -m shoplearn train --fee F --days N --seed S --out FILE [--resume FILE]
                          [--alpha A] [--half-life H] [--epsilon E]
```

| argument | meaning | allowed |
|---|---|---|
| `--fee` | delivery fee (`order_fixed_cost`) | integer ≥ 0 |
| `--days` | days to train **in this run** | integer ≥ 1 |
| `--seed` | seed of the run | integer ≥ 0 |
| `--alpha` | first step size, default 0.5 | in (0, 1] |
| `--half-life` | step-size half-life in visits, default 50 | > 0, `inf` allowed |
| `--epsilon` | exploration, default 0.2 | in [0, 1] |

- **Without `--resume`**, the table Q must be the one the loop of `run_q_learning` would produce for the same
  fee, seed, days and settings, using your three functions. Read that function in `lab08_core.py`: the seeds,
  the step-size schedule and the end of a 100-day episode all matter.
- **With `--resume`**, the program continues from the saved state for N more days and writes the new state to
  `--out` (which may be the same file). Training A days and then resuming for B days must give **the same
  table, bit for bit**, as training A + B days in one go, for any A and B.
- **Resuming must not replay the earlier days.** The point is that the program remembers.
- With `--resume`, `--fee` and `--seed` must still be given and must match the file. `--alpha`, `--half-life`
  and `--epsilon` are optional then (the saved values apply), and if given they must match. A mismatch is an
  error.

The file format of the checkpoint is your choice.

### evaluate

```
python -m shoplearn evaluate FILE
```

prints **exactly one JSON object** on standard output, with at least these keys:

| key | value |
|---|---|
| `days_trained` | total days over all sessions |
| `fee`, `seed` | the settings saved in the checkpoint |
| `score` | `score_policy` of the greedy policy of the table (`greedy_policy`: ties go to the smaller order) |
| `optimum` | the score of value iteration run with your `bellman_backup` |
| `regret` | `optimum - score` |
| `policy` | list of 11 integers: the order for stock 0, 1, …, 10 |
| `q_sha256` | `hashlib.sha256(np.ascontiguousarray(q, dtype="<f8").tobytes()).hexdigest()`, where `q` has shape (11, 11) and holds `-np.inf` for orders that do not fit |

### compare

```
python -m shoplearn compare --fee F --days N --n-seeds K
```

runs the three routes of Lab 07 with `compare_strategies` (Q-learning gets N days, the seeds are 0 … K−1 with
K ≥ 2, the sweep keeps its default) and prints one JSON object:

```json
{"fee": 200, "days": 3000, "n_seeds": 3, "optimum": 0.0,
 "routes": [{"route": "order-up-to sweep", "runs": 3, "regret_mean": 0.0, "score_sd": 0.0}]}
```

with one entry per route, named as in `Comparison.rows()` (the numbers above are placeholders).

### When something is wrong

In each of these cases the program **exits with a non-zero code, prints a short message on standard error that
says what is wrong, shows no Python traceback, and does not write `--out`**:

- a checkpoint that is missing, damaged (a truncated file, too) or of a format version your code does not read;
- an argument that is missing or outside the allowed range;
- a resume whose settings do not match the checkpoint.

### Same input, same output

The same command gives the same table every time.

## What your repository must contain

- **Your own `README.md`**, replacing this one (this text stays on the template repository). A stranger who
  follows it, from a fresh clone, must reach a trained and evaluated policy: give the exact commands, including
  the one that runs your tests.
- **Pinned dependencies**, installable from the repository root in one of two ways: `uv sync` (commit
  `uv.lock`; [uv projects](https://docs.astral.sh/uv/guides/projects/)) or `pip install -r requirements.txt`
  in a fresh virtual environment (exact `==` versions; add the line `-e .` if your layout needs your package
  installed). After either, `python -m shoplearn` (with uv: `uv run python -m shoplearn`) must work from the
  repository root.
- **Tests** that would catch a regression, above all in the part that remembers.
- **A CI workflow** that runs your tests on **every push**
  ([GitHub Actions quickstart](https://docs.github.com/en/actions/get-started/quickstart),
  [Python in GitHub Actions](https://docs.github.com/en/actions/tutorials/build-and-test-code/python)).
- **An incremental history** whose commit messages say what changed. Use branches or pull requests where they
  serve a purpose.
- **`DESIGN.md`**: the template in this repository, filled in.

A layout that works (not required):

```
README.md   DESIGN.md   pyproject.toml (or requirements.txt)   lab08_core.py
shoplearn/  __init__.py  __main__.py  learners.py  …
tests/
.github/workflows/
```

Do not commit your virtual environment, caches or checkpoint files.

## What you hand in

Submit in Canvas the URL of your **last commit** (`https://github.com/<you>/<repository>/commit/<sha>`). That
commit is the one that is marked. The repository must stay public until grades are posted. Late commits are
flagged, not hidden.

Your program is checked by running it against the contract above, from a fresh clone, by a script you do not
have. Your own tests are how you find out before that.

## What is assessed

The whole assignment (100 %) is evidence for ABET Performance Indicator **PI 2.2 Implementation**: *implements a
computing-based solution using appropriate tools, technologies, and programming practices*.

| Dimension | Weight | Points | Where the evidence is |
|---|---|---|---|
| Functional Implementation Against Requirements | 35 % | 5.25 | the program, run against the contract above |
| Version Control and Development Practice | 25 % | 3.75 | commit history, repository layout, pins, your `README.md` |
| Selection and Use of Tools and Technologies | 20 % | 3 | `DESIGN.md` §2 and how the code uses what you chose |
| Code Quality, Testing and Automation | 20 % | 3 | tests, CI runs on GitHub, error handling, readable code |

Each dimension is placed at one of four levels by the share of its marks you earn: **Exemplary** from 90 %,
**Satisfactory** from 70 %, **Developing** from 50 %, **Unsatisfactory** below 50 %. The dimensions, weights and
level descriptions are those of the ABET PI assessment document for CSS-4007, task HW5, p. 4. This lab is the
course's evidence for that indicator; its task is a reinforcement-learning one rather than the agent that
document describes. The 0 / 50 / 70 / 90 % reading of marks is still awaiting the coordinator's confirmation.

There is **no bonus** for extensions.

## AI use

AI tools, including coding agents, are allowed. Declare them in the last section of `DESIGN.md`: which tools,
for what, and which parts of the repository they produced. You must be able to explain every line you submit,
and every number you quote must be one your own program printed. The rules are on the course's AI use policy
page in Canvas.

## Glossary

- **Checkpoint**: a file holding everything a program needs to continue later as if it had never stopped.
- **Session**: one run of the program, from start to exit.
- **Deterministic**: the same inputs give the same outputs.
- **CI** (continuous integration): a service that runs your tests automatically on every push.
- **Pinned dependency**: a library recorded with its exact version.

## Files in this template

| File | What it is |
|---|---|
| `README.md` | The assignment (replace it with your own README) |
| `DESIGN.md` | The template you fill in |
| `lab08_core.py` | The shop, the three routes, the checks (supplied) |
| `.gitignore` | Keeps environments and caches out of the history |
