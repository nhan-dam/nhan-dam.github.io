---
title: GitHub Pages Setup
---

# 1. Overview

This note documents how to set up a personal website on GitHub Pages using MkDocs Material — a free, zero-maintenance way to publish technical notes alongside source code in the same repository.

The resulting site is served at `https://your-username.github.io/your-repo-name` with no domain name to buy or server to maintain. Every `git push` to `main` automatically rebuilds and redeploys the site via GitHub Actions.

---

# 2. Prerequisites

- A GitHub account.
- Git installed and configured locally (see Section 3).
- Python and `pip` available locally (only needed if you want to preview the site before pushing).

---

# 3. Git Configuration

Before creating any commits, configure git to use your GitHub private no-reply email address. This is required if the 'Block command line pushes that expose my email' privacy setting is enabled in GitHub (Settings → Emails) — pushes made with a real email address will be rejected.

Find your no-reply address at GitHub → Settings → Emails. It takes the form `123456789+your-username@users.noreply.github.com`.

```bash
git config --global user.name "Your Name"
git config --global user.email "123456789+your-username@users.noreply.github.com"
git config --global init.defaultBranch main
```

When generating an SSH key for GitHub authentication, use the same no-reply address as the key comment:

```bash
ssh-keygen -t ed25519 -C "123456789+your-username@users.noreply.github.com" -f ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
# → Add the output to GitHub: Settings → SSH Keys
```

Verify the remote URL uses SSH (not HTTPS) to avoid being prompted for a username and password on every push:

```bash
git remote -v
# Should show: git@github.com:your-username/your-repo.git
```

If it shows `https://`, switch to SSH:

```bash
git remote set-url origin git@github.com:your-username/your-repo.git
```

---

# 4. Repository Structure

The minimal structure required for MkDocs is:

```
your-repo/
├── .github/
│   └── workflows/
│       └── deploy.yml        ← GitHub Actions workflow
├── docs/                     ← all Markdown content goes here
│   ├── index.md              ← site homepage
│   └── javascripts/
│       └── mathjax.js        ← LaTeX rendering config
├── mkdocs.yml                ← site configuration
├── requirements.txt          ← Python dependencies
└── .gitignore
```

Notes and subdirectories live inside `docs/`. The `mkdocs.yml` file controls the site theme, navigation, and plugins. Everything else is generated automatically and should not be committed.

Add the following to `.gitignore` to exclude the local build output:

```
site/
__pycache__/
*.pyc
.DS_Store
```

---

# 5. Configuration Files

## 5.1. `requirements.txt`

```
mkdocs-material>=9.5
```

## 5.2. `mkdocs.yml`

```yaml
site_name: Your Site Name
site_url: https://your-username.github.io/your-repo-name

repo_name: your-username/your-repo-name
repo_url: https://github.com/your-username/your-repo-name
edit_uri: edit/main/docs/

theme:
  name: material
  palette:
    - scheme: default
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - scheme: slate
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  features:
    - navigation.tabs
    - navigation.tabs.sticky
    - navigation.sections
    - navigation.top
    - search.highlight
    - search.suggest
    - content.code.copy

plugins:
  - search

markdown_extensions:
  - pymdownx.arithmatex:
      generic: true
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.superfences
  - admonition
  - toc:
      permalink: true

extra_javascript:
  - javascripts/mathjax.js
  - https://unpkg.com/mathjax@3/es5/tex-mml-chtml.js

nav:
  - Home: index.md
  - Section One:
    - Page A: section-one/page-a.md
```

The `nav` block controls the sidebar navigation. Each entry maps a label to a Markdown file path relative to `docs/`.

## 5.3. `docs/javascripts/mathjax.js`

Required for LaTeX rendering in notes. Without this file, the `extra_javascript` entry in `mkdocs.yml` will have no effect.

```javascript
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};
```

## 5.4. `.github/workflows/deploy.yml`

This workflow triggers on every push to `main`, installs MkDocs Material, and deploys the built site to the `gh-pages` branch.

```yaml
name: Deploy MkDocs to GitHub Pages

on:
  push:
    branches:
      - main

permissions:
  contents: write

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4.2.2
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5.6.0
        with:
          python-version: '3.x'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Deploy to GitHub Pages
        run: mkdocs gh-deploy --force
```

> **Action versions:** `actions/checkout@v4.2.2` and `actions/setup-python@v5.6.0` are the minimum versions required for Node.js 24 compatibility on GitHub Actions runners. GitHub is deprecating Node.js 20 for actions from June 2026 and removing it entirely in September 2026. Using these pinned versions avoids deprecation warnings in the build log.

> **Residual warning:** A warning about `actions/upload-artifact@v4` may still appear. This action is called internally by `mkdocs gh-deploy` and is not directly controllable from this workflow file. It is safe to ignore until a future release of `mkdocs-material` updates their tooling.

---

# 6. GitHub Repository Settings

## 6.1. Create the Repository

Create a new repository on GitHub. The site will be served at `https://your-username.github.io/your-repo-name`, so choose the repo name accordingly.

## 6.2. Push the Initial Commit

```bash
cd your-repo
git init
git add .
git commit -m "Initial commit"
git remote add origin git@github.com:your-username/your-repo-name.git
git push -u origin main
```

If GitHub auto-created files (e.g. a `LICENSE` or `README.md`) when the repo was made, the push will be rejected with a 'fetch first' error. Pull first to merge those files, then push:

```bash
git pull origin main --rebase
git push origin main
```

> **`--rebase` vs plain `git pull`:** Without `--rebase`, `git pull` creates an extra merge commit joining the two histories. With `--rebase`, your local commits are replayed on top of the remote commits, producing a clean linear history. For a personal repo this is mostly aesthetic, but it is a good habit.

## 6.3. Configure GitHub Pages

After the first push, the Actions workflow will run and create a `gh-pages` branch. Once it completes:

1. Go to the repository on GitHub.
2. Click **Settings** → **Pages**.
3. Under **Source**, select **Deploy from a branch**.
4. Choose branch **`gh-pages`**, folder **`/ (root)`**.
5. Click **Save**.

The site will be live within about 30 seconds at `https://your-username.github.io/your-repo-name`.

> **GitHub Actions vs Deploy from a branch:** The Pages source setting has two options. 'GitHub Actions' delegates the entire build and deploy process to a workflow. 'Deploy from a branch' serves static files directly from a branch. MkDocs uses the latter: the `mkdocs gh-deploy` command builds the site locally (in the Actions runner) and pushes the output to the `gh-pages` branch, which GitHub Pages then serves. So the correct setting is 'Deploy from a branch', not 'GitHub Actions'.

---

# 7. Ongoing Workflow

Every subsequent `git push` to `main` triggers a rebuild automatically. No manual steps are needed after the initial setup.

To force a rebuild without making a file change (e.g. to test the workflow):

```bash
git commit --allow-empty -m "Force rebuild"
git push origin main
```

To preview the site locally before pushing:

```bash
pip install -r requirements.txt
mkdocs serve
# Open http://localhost:8000
```

---

# 8. URL Structure

| Repo name | Site URL |
|---|---|
| `your-username.github.io` | `https://your-username.github.io` |
| any other name (e.g. `ml-notes`) | `https://your-username.github.io/ml-notes` |

A repo named exactly `your-username.github.io` produces a root personal site. Any other repo name produces a project site at a sub-path.

---

# 9. Troubleshooting

## 9.1. Site shows raw `README.md` instead of the MkDocs site

The `gh-pages` branch has not been created yet, or GitHub Pages is not pointed at it. Check that the Actions workflow has run successfully (repository → **Actions** tab), then verify the Pages source setting as described in Section 6.3.

## 9.2. Old version of the site appears after deployment

This is a browser caching issue. Hard-refresh to bypass the cache:

- **Mac:** `Cmd + Shift + R`.
- **Windows/Linux:** `Ctrl + Shift + R`.

Alternatively, open the site in a private/incognito window, which has no cache. To permanently clear the cache for the site in Chrome: DevTools (`Cmd + Option + I`) → **Application** → **Storage** → **Clear site data**.

## 9.3. Push rejected with email privacy error

GitHub rejects pushes that expose a real email address when the privacy setting is enabled. Fix by amending the commit author to use the no-reply address:

```bash
git commit --amend --reset-author
git push origin main --force
```

If the offending commits have already been squashed or rebased, the same `--amend --reset-author` approach applies to whichever commit is at `HEAD`.
