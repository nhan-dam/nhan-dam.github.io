# GitHub Task-Oriented Cheatsheet

> Created on: 20 April 2026
>
> Updated on: 6 August 2026

## 1. Create a New Local Repository

**Note for `uv` projects:** `uv init` automatically runs `git init` and creates a `.gitignore` pre-populated with `.venv/` and `.python-version`. If you are scaffolding a project with `uv`, skip `git init`, since the repository already exists.

For a plain directory not managed by `uv`:

```bash
git init
git add .
git commit -m "Initial commit"
```

---

## 2. Push a Local Repository to a New Remote Repository

This procedure links a local repository holding at least one commit (see [Section 1](#1-create-a-new-local-repository)) to a newly created remote, pushes it, and configures the licence.

### 2.1. Create the Remote Repository Without Auto-Initialisation

Create the repository on GitHub with no README, no `.gitignore`, and no licence template selected. Each of those options writes a commit to the remote that shares no ancestor with the local history, which makes the first push fail and forces the reconciliation of [Section 2.5](#25-reconcile-a-rejected-first-push). The licence is added after the first push instead, as described in [Section 2.4](#24-add-the-licence).

### 2.2. Audit What the First Commit Would Contain

Git retains every version of every blob permanently, so one oversized file inflates all future clones even after it is deleted, and GitHub rejects any file above 100 MB. Machine-learning repositories are especially exposed, since model weights, checkpoints, merged models, and exported artefacts accumulate in the working tree. Verify the ignore rules before the first commit rather than after:

```bash
git ls-files -oc --exclude-standard | wc -l   # number of files that would be committed
du -sh --exclude=.git .                       # working-tree size, ignore rules not applied
```

If the two disagree by orders of magnitude, the ignore rules are incomplete. Typical entries for a training repository are the results directory, any `*-model` and `*-model-merged` directories, `*.gguf`, and `.DS_Store`.

Large artefacts belong elsewhere. Weights go to a model registry, e.g. the Hugging Face Hub. Datasets and checkpoints go to object storage tracked by DVC or git-annex, which keeps a small pointer file in Git. Only small, human-readable artefacts that documents cite, e.g. metric summaries and evaluation tables, are worth committing directly.

### 2.3. Link and Push

```bash
git remote add origin git@github.com:<user>/<repo>.git
git branch -M main
git push -u origin main
```

- `git branch -M main` renames the current branch to `main`, forcing the rename even if a branch called `main` already exists (`-M` is shorthand for `--move --force`). This is necessary because `git init` may create a default branch named `master` depending on the Git version or local configuration.
- `-u` sets the upstream tracking reference so that subsequent `git push` and `git pull` calls require no additional arguments.
- `git remote -v` confirms the remote resolved as intended before pushing.

If the remote was already added with an HTTPS URL (which prompts for credentials even when an SSH key is configured), switch it to SSH:

```bash
git remote set-url origin git@github.com:<user>/<repo>.git
```

### 2.4. Add the Licence

Add the licence through the GitHub interface after the first push, so that the template text, the copyright year, and the owner name are filled in automatically. Select 'Add file', then 'Create new file', name the file `LICENSE`, and use the 'Choose a licence template' button that appears. Commit to `main`, then bring the commit down:

```bash
git pull
```

That commit is a child of the pushed history, so the pull is a fast-forward and leaves no merge commit. Recording the licence in the packaging metadata as well keeps the built artefact consistent with the repository, e.g. `license = "MIT"` in `pyproject.toml` for a Python project.

> **Check every licence, not only the repository's own.** The chosen licence covers the code and prose written for the repository. It does not cover anything the repository consumes or redistributes. Datasets, base models, and vendored code keep their own terms, some of which impose obligations such as attribution or share-alike. Before publishing, enumerate every external asset the repository ships, quotes, or documents, confirm that redistribution is permitted, and record each licence in the README alongside the repository's own. Verbatim excerpts, e.g. sample rows printed into a committed exploratory-data-analysis report, remain under their upstream terms regardless of the repository licence.

### 2.5. Reconcile a Rejected First Push

If the push is rejected with a 'fetch first' error, the remote contains commits not present locally, e.g. a `LICENSE` or `README.md` auto-created by GitHub. Pull with rebase to replay the local commits on top of the remote ones, then push:

```bash
git pull origin main --rebase
git push -u origin main
```

Using `--rebase` produces a clean linear history. Without it, `git pull` inserts an extra merge commit joining the two histories.

When the two histories have no commit in common, which is the usual case for a repository initialised on both sides, Git refuses outright with 'refusing to merge unrelated histories'. Add the override:

```bash
git pull origin main --rebase --allow-unrelated-histories
```

---

## 3. Set Up Continuous Integration

Continuous integration (CI) runs the test suite automatically on every push. GitHub Actions executes any workflow file found under `.github/workflows/`, so committing the file is the only setup required.

### 3.1. Add the Workflow

The following workflow tests a `uv`-managed Python project. Save it as `.github/workflows/ci.yml`.

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:

jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
          cache-dependency-glob: uv.lock
      - run: uv python install
      - run: uv sync --locked
      - run: uv run pytest -q
```

- `workflow_dispatch` adds a 'Run workflow' button to the Actions tab, which starts the workflow on demand.
- `uv python install` reads `.python-version`, so the runner and the workstation agree on the interpreter.
- `uv sync --locked` fails when `uv.lock` is out of step with `pyproject.toml`, so a stale lockfile becomes a failed build rather than a silent re-resolution.
- `enable-cache` with `cache-dependency-glob: uv.lock` reuses the dependency cache until the lockfile changes.

> **Expect a delay on the first run.** The first push-triggered run on a new repository can take tens of minutes to be scheduled, and the Actions tab reports no runs in the meantime. Subsequent runs queue immediately. Use the 'Run workflow' button to verify the workflow without waiting.

### 3.2. Add the Status Badge

Place the badge directly beneath the README title, separated from it by a blank line.

```markdown
[![CI](https://github.com/<user>/<repo>/actions/workflows/ci.yml/badge.svg)](https://github.com/<user>/<repo>/actions/workflows/ci.yml)
```

- The badge reports the most recent run of that workflow on the default branch. Wrapping it in a link opens the Actions tab.
- Append `?branch=main` to pin the badge to one branch, or `?event=push` to ignore manually dispatched runs.
- On a private repository the badge image does not render for viewers without access, so it is worth adding when the repository is made public.

---

## 4. Create a Local Branch

Create a new branch and switch to it immediately.

```bash
git switch -c <branch-name>
```

The `-c` flag (shorthand for `--create`) creates the branch and switches to it in one step. It is equivalent to the older `git checkout -b <branch-name>`.

To create the branch without switching to it:

```bash
git branch <branch-name>
```

---

## 5. Show All Local and Remote Branches

List local branches only:

```bash
git branch
```

List remote branches only:

```bash
git branch -r
```

List all local and remote branches together:

```bash
git branch -a
```

---

## 6. Delete a Local or Remote Branch

Delete a local branch, i.e. the safe form, which refuses if the branch has unmerged changes:

```bash
git branch -d <branch-name>
```

Force-delete a local branch regardless of merge status:

```bash
git branch -D <branch-name>
```

Delete a remote branch:

```bash
git push origin --delete <branch-name>
```

---

## 7. Sync the Branch List Between Local and Remote

Remote branches that have been deleted on the server are not automatically removed from the local reference list. To prune stale remote-tracking references:

```bash
git fetch --prune
```

To make pruning the default behaviour on every fetch, set it once globally:

```bash
git config --global fetch.prune true
```

---

## 8. Squash Commits

'Squashing' collapses multiple commits into a single commit. Before squashing, inspect the recent commit log to determine how many commits to include:

```bash
git log --oneline -<n>
```

By default, `git log --oneline` prints all commits (paginated via `less`, so press `q` to quit). Passing `-<n>` limits the output to the `<n>` most recent commits. For most squash decisions, `-10` or `-20` is sufficient, e.g.:

```bash
git log --oneline -10
```

The output shows one line per commit with the short hash and subject message, e.g.:

```
e3f1a2b Fix tokeniser edge case
9c4d7f0 Add reward model training loop
3b8a1c5 Initial dataset preprocessing
```

Once you have identified the target range, run an interactive rebase where `<n>` is the number of commits to include:

```bash
git rebase -i HEAD~<n>
```

In the editor that opens, leave the first entry as `pick` and change the remaining entries to `squash` (or `s`). Save and close. Git then prompts for a combined commit message.

**Note:** squashing rewrites history, so if any of the squashed commits were already on the remote, a normal `git push` will be rejected because the local `HEAD` is now behind the remote. Force-push to overwrite the remote history:

```bash
git push --force-with-lease
```

Prefer `--force-with-lease` over `--force`. It refuses the push if the remote has received commits since the last fetch, guarding against accidentally overwriting others' work. Coordinate with collaborators before force-pushing to any shared branch.

---

## 9. Permanently Discard Current Changes and Revert to a Prior Commit

First, identify the target commit hash using the one-line log, which is also useful for counting commits to squash (see [Section 8](#8-squash-commits)):

```bash
git log --oneline -<n>
```

Copy the short hash of the desired target commit, then perform a hard reset. This moves `HEAD` to the specified commit and discards all subsequent commits and uncommitted changes. The operation is irreversible.

```bash
git reset --hard <commit-hash>
```

If the branch has already been pushed to a remote, force-push to overwrite the remote history:

```bash
git push --force-with-lease
```

Prefer `--force-with-lease` over `--force`. It refuses the push if the remote has received commits since the last fetch, guarding against accidentally overwriting others' work.

---

## 10. Temporarily Discard Current Changes and Revert to a Prior Commit

'Stashing' saves uncommitted changes to a temporary stack and restores a clean working directory, without permanently discarding anything.

```bash
git stash push -m "<description>"
git reset --hard <commit-hash>
```

To restore the stashed changes later:

```bash
git stash pop
```

`git stash pop` applies the most recent stash entry and removes it from the stack. To apply without removing it, use `git stash apply` instead.

---

## 11. Permanently Discard Current Changes and Switch Branch

Discard all uncommitted changes in the working directory and index, then switch to another branch. This is irreversible.

```bash
git reset --hard HEAD
git switch <branch-name>
```

Or, equivalently, in a single step using the `--discard-changes` flag:

```bash
git switch --discard-changes <branch-name>
```

---

## 12. Temporarily Discard Current Changes and Switch Branch

Stash uncommitted changes before switching, preserving them for later retrieval.

```bash
git stash push -m "<description>"
git switch <branch-name>
```

To bring the stashed changes back after returning to the original branch:

```bash
git switch <original-branch>
git stash pop
```
