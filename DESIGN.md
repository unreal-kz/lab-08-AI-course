# Design notes

<!-- STUDENT: replace -->

Replace the guidance below with your own text and **delete the line above**; a file that still carries it counts
as not written. Short and specific beats long. Every number must be one your own program printed.

## 1. What the program remembers, and what it recomputes

A table with one row per piece of information the program needs to continue or to answer `evaluate`:

| Information | Saved in the checkpoint, or recomputed? | Why |
|---|---|---|
| | | |

Then: your program is deterministic, so you could save only the fee, the seed and the number of days and replay
them. Why do you not? What would replaying cost, and what would break it?

## 2. Tools and storage

For each of these: what you chose, one alternative you considered, and why you chose as you did.

- Checkpoint format
- Command line library
- Test tool
- Dependency pinning
- CI

## 3. Failure behaviour

| Failure | What the program does | Which test covers it |
|---|---|---|
| | | |

Name one failure your program does **not** handle, and what handling it would take.

## 4. AI-use declaration

Which AI tools you used, for what, and which parts of the repository they produced. If you used none, say so.
