# GitHub Task-Oriented Cheatsheet

> Created on: 20 April 2026
>
> Updated on: 28 April 2026

## 1. Create a New Local Repository

**Note for `uv` projects:** `uv init` automatically runs `git init` and creates a `.gitignore` pre-populated with `.venv/` and `.python-version`. If you are scaffolding a project with `uv`, skip `git init` — the repository already exists.

For a plain directory not managed by `uv`:

```bash
git init
git add .
git commit -m "Initial commit"
```

---

## 2. Push a Local Repository to a New Remote Repository

Link the local repository to a newly created remote (e.g. on GitHub), then push all commits.

```bash
git remote add origin https://github.com/<user>/<repo>.git
git branch -M main
git push -u origin main
```

- `git branch -M main` renames the current branch to `main`, forcing the rename even if a branch called `main` already exists (`-M` is shorthand for `--move --force`). This is necessary because `git init` may create a default branch named `master` depending on the Git version or local configuration.
- `-u` sets the upstream tracking reference so that subsequent `git push` and `git pull` calls require no additional arguments.

---

## 3. Create a Local Branch

Create a new branch and switch to it immediately.

```bash
git switch -c <branch-name>
```

The `-c` flag (shorthand for `--create`) creates the branch and switches to it in one step; it is equivalent to the older `git checkout -b <branch-name>`.

To create the branch without switching to it:

```bash
git branch <branch-name>
```

---

## 4. Show All Local and Remote Branches

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

## 5. Delete a Local or Remote Branch

Delete a local branch (safe — refuses if the branch has unmerged changes):

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

## 6. Sync the Branch List Between Local and Remote

Remote branches that have been deleted on the server are not automatically removed from the local reference list. To prune stale remote-tracking references:

```bash
git fetch --prune
```

To make pruning the default behaviour on every fetch, set it once globally:

```bash
git config --global fetch.prune true
```

---

## 7. Squash Commits

'Squashing' collapses multiple commits into a single commit. Before squashing, inspect the recent commit log to determine how many commits to include:

```bash
git log --oneline -<n>
```

By default, `git log --oneline` prints all commits (paginated via `less`; press `q` to quit). Passing `-<n>` limits the output to the `<n>` most recent commits. For most squash decisions, `-10` or `-20` is sufficient, e.g.:

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

In the editor that opens, leave the first entry as `pick` and change the remaining entries to `squash` (or `s`). Save and close; Git will then prompt for a combined commit message.

**Note:** squashing rewrites history, so if any of the squashed commits were already on the remote, a normal `git push` will be rejected because the local `HEAD` is now behind the remote. Force-push to overwrite the remote history:

```bash
git push --force-with-lease
```

Prefer `--force-with-lease` over `--force`; it refuses the push if the remote has received commits since the last fetch, guarding against accidentally overwriting others' work. Coordinate with collaborators before force-pushing to any shared branch.

---

## 8. Permanently Discard Current Changes and Revert to a Prior Commit

First, identify the target commit hash using the one-line log (also useful for counting commits to squash; see Section 7):

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

Prefer `--force-with-lease` over `--force`; it refuses the push if the remote has received commits since the last fetch, guarding against accidentally overwriting others' work.

---

## 9. Temporarily Discard Current Changes and Revert to a Prior Commit

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

## 10. Permanently Discard Current Changes and Switch Branch

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

## 11. Temporarily Discard Current Changes and Switch Branch

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

---

## 12. Add `.gitignore` to an Existing Repository

Adding a `.gitignore` after the initial commit requires an extra step. Git only respects `.gitignore` rules for **untracked** files; files that were already committed remain tracked regardless of any ignore rules added later. To untrack them, the Git index (i.e. the staging area) must be cleared and rebuilt.

### 12.1. Create the `.gitignore` file

Create `.gitignore` in the repository root and add the desired patterns, one per line. For example:

```
__pycache__/
*.pyc
.env
.venv/
```

### 12.2. Clear the index and re-stage all files

Remove all files from the index without deleting them from the local working directory, then re-add everything so that the new `.gitignore` rules are applied:

```bash
git rm -r --cached .
git add .
```

`git rm --cached` removes a file from the index only, leaving the local copy intact. The `-r` flag (i.e. recursive) is required when the target is a directory (`.` refers to the entire repository root). After re-running `git add .`, any path matching a pattern in `.gitignore` is excluded from the index.

### 12.3. Commit and push

```bash
git commit -m "Add .gitignore and untrack ignored files"
git push
```

**Note:** if only specific files need to be untracked rather than the entire repository, replace the broad `git rm -r --cached .` with a targeted call:

```bash
git rm --cached <path/to/file>
```

This avoids unnecessarily re-staging every file in the repository.
