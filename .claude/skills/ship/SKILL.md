---
name: ship
description: Run the project's standard implement, test, review pipeline for a task using the developer, unit-tester and reviewer agents.
argument-hint: <task description>
---

Run the project's standard pipeline for this task: $ARGUMENTS

If no task was given, ask the user what to ship.

## Steps

1. **Implement.** Launch the `developer` agent with the task, relevant context, and any constraints. Wait for its summary and the list of changed files.

2. **Test and review.** Launch `unit-tester` and `reviewer` **in parallel** (one message, two Agent calls).
   - `unit-tester`: give it the developer's summary and changed files; ask it to write/update tests and run the suite.
   - `reviewer`: give it the goal and ask it to review the current diff (`git diff`), including untracked files.

3. **Fix loop** (max 2 rounds). If `reviewer` says **Request changes**, or `unit-tester` reports failing tests that reveal a real bug:
   - Send the exact findings to `developer` to fix.
   - Re-run `unit-tester` and `reviewer` in parallel.
   - If still not approved after 2 rounds, stop and report what remains.

4. **Report** to the user in a short summary:
   - What changed (files, key decisions)
   - Tests added and their results
   - Reviewer's final verdict and any unresolved findings
   - Anything skipped or that couldn't be verified

## Rules

- Pass context explicitly in every delegation; subagents don't share reasoning.
- Agents that edit files must not run in parallel with each other. Only `unit-tester` and `reviewer` run together.
- Do not commit or push. Leave changes in the working tree for the user.
- Report failures and skipped steps plainly.
