# ML Workstation Setup: Mac Mini M4 Pro with uv

> Created on: 3 April 2026
>
> Updated on: 4 May 2026

**Machine:** Mac mini M4 Pro · 64 GB RAM · internal 512 GB SSD.

**External drive:** Samsung 990 Pro 2 TB in OWC M.2 Express (Thunderbolt/USB4).

**Purpose:** set up a workstation for ML projects (e.g. Kaggle challenges and RLHF).

---

## 1. Philosophy

The storage split between drives is as follows.

- **Internal SSD** → macOS, Homebrew, `uv` CLI (~10 MB), shell config, VS Code extensions.
- **External NVMe** → Python installs, virtual environments, all caches, model weights, datasets, source code, Ollama models.

`uv` uses per-project virtual environments, i.e. each project or competition has its own isolated `.venv`, stored inside the project folder on the external drive.

---

## 2. Prepare the External Drive

### 2.1. Format as APFS

1. Open **Disk Utility** (Spotlight → 'Disk Utility').
2. Select the Samsung 990 Pro (the physical drive, not a partition).
3. Click **Erase** and fill in the following fields.
   - **Name:** `ML_Workspace`.
   - **Format:** `APFS`.
   - **Scheme:** `GUID Partition Map`.
4. Click **Erase** to confirm.

> **Why APFS?** Native macOS filesystem, supporting snapshots, efficient space sharing, and fast metadata operations. Better than exFAT for a Mac-only workflow.
>
> **Why `ML_Workspace` and not `ML Workspace`?** Spaces in volume names require quoting or escaping in every shell command (`/Volumes/ML\ Workspace`), which is constant friction. An underscore is equally readable with zero shell complexity.

### 2.2. Verify Mount Point

```bash
ls /Volumes/ML_Workspace
```

On a freshly formatted empty volume you will see either blank output (normal) or a `.Spotlight-V100` folder that macOS creates automatically for search indexing. Both are fine.

```bash
diskutil info /Volumes/ML_Workspace | grep "Volume Name"
```

Expected output:

```
Volume Name:               ML_Workspace
```

Key fields to confirm in the full `diskutil info` output for a broader sanity check.

- `Device Location: External` — confirms it is the external drive, not the internal SSD.
- `Solid State: Yes` and `Protocol: PCI-Express` — confirms it is the NVMe.
- `SMART Status: Verified` — drive health is good.
- `Volume Used Space:` — should be near zero after a fresh format.

### 2.3. Verify Auto-Mount on Login

Reboot once and confirm `/Volumes/ML_Workspace` is present. APFS volumes auto-mount on macOS by default.

---

## 3. Directory Structure on External Drive

```bash
mkdir -p /Volumes/ML_Workspace/{projects,datasets,models,kaggle}
mkdir -p /Volumes/ML_Workspace/python          # uv-managed Python installs
mkdir -p /Volumes/ML_Workspace/cache/{uv,pip,huggingface,torch}
mkdir -p /Volumes/ML_Workspace/ollama/models
```

Final structure:

```
/Volumes/ML_Workspace/
├── projects/               ← git repos, course notebooks (each has its own .venv)
│   ├── rlhf-course/
│   │   └── .venv/          ← project-specific venv (created by uv)
│   └── kaggle-titanic/
│       └── .venv/
├── datasets/               ← raw datasets (non-Kaggle)
├── models/                 ← manually downloaded weights
├── python/                 ← uv-managed Python versions (3.12, 3.11, etc.)
├── cache/
│   ├── uv/                 ← uv package cache (shared across all projects)
│   ├── pip/                ← fallback pip cache
│   ├── huggingface/        ← HF Hub: models, tokenizers, datasets
│   └── torch/              ← torch.hub cache
├── kaggle/                 ← Kaggle competition data
└── ollama/
    └── models/             ← Ollama model blobs
```

> **Key insight:** Each project's `.venv` lives inside the project folder on the external drive. The `uv` cache at `/Volumes/ML_Workspace/cache/uv` is shared across all projects via hard links — if two projects need the same PyTorch wheel, it is downloaded once and reused. This is much more space-efficient than conda's approach.

---

## 4. Core System Tools

### 4.1. Check for Xcode Command Line Tools

The Xcode Command Line Tools provide the compiler toolchain (`clang`, `make`) that Homebrew and some Python C-extension builds depend on. Check if already installed:

```bash
xcode-select -p
```

- If it returns a path like `/Library/Developer/CommandLineTools` — already installed, skip to [Section 4.2](#42-install-homebrew).
- If it returns an error — install it:

```bash
xcode-select --install
```

Accept the dialog. Takes approximately five minutes.

### 4.2. Install Homebrew

Homebrew is a package manager for macOS, letting you install command-line tools with a single command rather than downloading installers manually. It is the standard way to get system-level tools that are not Python packages and cannot be installed via `uv`.

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

After installation, Homebrew will print two commands to run. Run both exactly as printed — they will look like:

```bash
# 1. Adds a blank line to ~/.zprofile for readability
echo >> ~/.zprofile

# 2. Adds Homebrew to your PATH permanently
echo 'eval "$(/opt/homebrew/bin/brew shellenv zsh)"' >> ~/.zprofile
```

Then apply the changes to your current terminal session without restarting it:

```bash
eval "$(/opt/homebrew/bin/brew shellenv zsh)"
```

> **Why `~/.zprofile` and not `~/.zshrc`?**
> - `~/.zprofile` runs once at login, the right place for PATH setup.
> - `~/.zshrc` runs on every new shell instance, the right place for aliases and environment variables (used in [Section 6](#6-shell-environment-variables)).
>
> Homebrew intentionally targets `~/.zprofile` for this reason. Everything else in this guide goes into `~/.zshrc`.

Verify:

```bash
brew --version
```

### 4.3. Install System Dependencies via Homebrew

```bash
brew install git wget htop tmux tree swig cmake
```

- `git` — version control.
- `wget` — download files from the terminal (complements macOS's built-in `curl`).
- `htop` — interactive CPU and RAM monitor.
- `tmux` — terminal multiplexer, keeps training runs alive if your terminal closes.
- `tree` — prints a visual directory structure (e.g. `tree /Volumes/ML_Workspace/projects`).
- `swig` — required by some Gymnasium environments (e.g. Box2D).
- `cmake` — required by some ML C-extension builds.

> **Note on `curl`:** macOS ships with `curl` pre-installed and it is sufficient for all ML work. Adding Homebrew's `curl` to this list generates a 'keg-only' warning — Homebrew installs it but deliberately keeps it out of your PATH to avoid conflicting with the system version. It adds noise without benefit, so leave it out.

---

## 5. Install `uv`

`uv` is a Rust-based Python package and environment manager. It replaces `pyenv`, `virtualenv`, and `pip` in one tool and is 10–100× faster than pip/conda for package installation.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

> **What this command does:**
> - `curl -LsSf` — downloads the install script from Astral's server (`-L` follows redirects, `-s` suppresses progress output, `-S` still shows errors, `-f` fails cleanly on server errors).
> - `|` — the pipe operator: feeds the downloaded content directly to the next command without saving to disk.
> - `sh` — executes the script immediately.
>
> The script installs the `uv` binary to `~/.local/bin/uv` and appends a PATH entry for `~/.local/bin` to `~/.zshrc`. The script is never written to disk — it flows through memory and is gone once `sh` finishes executing it.

Apply the new PATH entry to your current session:

```bash
source ~/.zshrc
```

> **Why `source ~/.zshrc`?** The installer wrote a new line to `~/.zshrc` on disk, but your current terminal session loaded that file before the line existed. `source` replays the file in your current session, making `uv` immediately available without opening a new terminal window.

Verify:

```bash
uv --version
which uv
# ~/.local/bin/uv
```

---

## 6. Shell Environment Variables

This phase redirects all ML tools to store their files on the external NVMe. Add all path configuration to `~/.zshrc` in one block:

```bash
cat >> ~/.zshrc << 'EOF'

# ── ML Workstation: External NVMe Paths ──────────────────────────────────────
export ML_ROOT="/Volumes/ML_Workspace"

# uv: Python installs and package cache on external drive
export UV_PYTHON_INSTALL_DIR="$ML_ROOT/python"
export UV_CACHE_DIR="$ML_ROOT/cache/uv"

# HuggingFace Hub (models, tokenizers, datasets)
export HF_HOME="$ML_ROOT/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$ML_ROOT/cache/huggingface/hub"

# PyTorch Hub
export TORCH_HOME="$ML_ROOT/cache/torch"

# pip cache (fallback)
export PIP_CACHE_DIR="$ML_ROOT/cache/pip"

# Ollama models
export OLLAMA_MODELS="$ML_ROOT/ollama/models"

# Ollama performance settings (Apple Silicon optimisations)
export OLLAMA_FLASH_ATTENTION="1"
export OLLAMA_KV_CACHE_TYPE="q8_0"

# Kaggle credentials
export KAGGLE_USERNAME="YOUR_KAGGLE_USERNAME"
export KAGGLE_API_TOKEN="YOUR_API_TOKEN_HERE"

# Convenience aliases
alias cdml="cd $ML_ROOT"
alias cddata="cd $ML_ROOT/datasets"
# ─────────────────────────────────────────────────────────────────────────────
EOF

source ~/.zshrc
```

> **How the `<< 'EOF'` block works:**
> - `cat >> ~/.zshrc` — appends content to `~/.zshrc` (`>>` appends; `>` would overwrite).
> - `<< 'EOF'` — tells the shell to read everything that follows as input, until it sees `EOF` on a line by itself.
> - The single quotes around `'EOF'` are important: they prevent the shell from expanding variables like `$ML_ROOT` before writing them to the file. You want the literal variable references written to `~/.zshrc` so they are evaluated fresh each time a terminal opens.
> - The closing `EOF` signals the end of the block.
>
> **`OLLAMA_FLASH_ATTENTION` and `OLLAMA_KV_CACHE_TYPE`** are Apple Silicon memory optimisations for Ollama.
> - `OLLAMA_FLASH_ATTENTION="1"` — enables Flash Attention, reducing memory usage during inference.
> - `OLLAMA_KV_CACHE_TYPE="q8_0"` — quantises the KV cache (i.e. the temporary attention scratch space) to 8-bit instead of 16-bit, cutting its memory footprint roughly in half. This does **not** affect model weight precision, only the intermediate computation buffer. Output quality difference is imperceptible in practice.
>
> **`KAGGLE_USERNAME` and `KAGGLE_API_TOKEN`** — Kaggle's current authentication method. The CLI reads these environment variables directly; no credentials file on disk is needed. Replace the placeholder values with your actual username and API token from kaggle.com → Settings → API.
>
> **Do this before installing any Python or packages.** These environment variables ensure `uv` downloads Python and caches wheels to the external drive from the very first install.

---

## 7. Install Python via `uv`

`uv` manages its own Python downloads, with no need for `pyenv` or Homebrew Python.

```bash
# Python 3.12 — recommended for Apple Silicon ML as of April 2026
uv python install 3.12

# Python 3.11 — keep as a fallback for any niche compatibility edge case
uv python install 3.11

# Verify both installed to the external drive
uv python list
ls /Volumes/ML_Workspace/python/
```

> **Why 3.12 over 3.11?** As of April 2026, Python 3.12 native arm64 wheels are available for every package in this stack — PyTorch (MPS), MLX, TRL, PEFT, Gymnasium — with no workarounds. The mlx-lm documentation explicitly targets Python 3.12 for Apple Silicon `uv` setups. Python 3.11 is kept only as a fallback in case a niche compatibility issue arises with a specific library.
>
> **One known limitation regardless of Python version:** Some PyTorch MPS operators still fall back to CPU on Apple Silicon. This is an Apple Metal constraint, not a Python version issue — 3.12 and 3.11 are equally affected.
>
> **Note on Ollama's bundled Python:** Ollama installs its own Python runtime (currently 3.14) internally for its own tooling. This is completely isolated from your `uv`-managed Pythons and not added to your PATH. It has no effect on your projects.

---

## 8. Project-Based Workflow with `uv`

`uv` encourages per-project virtual environments. This is better for ML work: your RLHF course and a Kaggle competition can pin different library versions without conflict.

### 8.1. Create a New Project

```bash
mkdir /Volumes/ML_Workspace/projects/rlhf-course
cd /Volumes/ML_Workspace/projects/rlhf-course
uv init --python 3.12
```

Verify the project structure was created correctly, including hidden files:

```bash
tree -a -L 1 /Volumes/ML_Workspace/projects/rlhf-course
```

Expected output:

```
rlhf-course/
├── .python-version   ← pins Python 3.12 for this project
├── .venv/            ← created on first `uv add` or `uv sync`
├── pyproject.toml    ← dependency manifest
└── README.md
```

> **Why `tree -a -L 1`?**
> - `-a` includes hidden files and directories (those starting with `.`) which are not shown by default.
> - `-L 1` limits output to one level deep, so you see the project root contents without being flooded by files inside `.venv` and other subdirectories.

### 8.2. Add Packages

```bash
cd /Volumes/ML_Workspace/projects/rlhf-course

# Core ML stack
uv add torch torchvision torchaudio

# MLX (Apple Silicon native — zero-copy unified memory, faster inference than PyTorch MPS)
uv add mlx mlx-lm

# HuggingFace + RLHF stack
uv add transformers datasets tokenizers accelerate peft trl evaluate huggingface_hub

# RL environments
uv add "gymnasium[all]"

# Experiment tracking
uv add wandb tensorboard

# Data science
uv add numpy pandas scipy scikit-learn matplotlib seaborn plotly tqdm rich einops

# Jupyter
uv add jupyterlab ipywidgets ipykernel
```

> **Why packages are grouped rather than combined into one command:** grouping by purpose makes it clear which package belongs to which role. Each `uv add` group resolves and installs immediately, with packages cached in `/Volumes/ML_Workspace/cache/uv` and hard-linked into `.venv`.
>
> **Note on PyTorch size on Apple Silicon:** PyTorch installs much smaller than on Linux (~400 MB vs ~2 GB) because the macOS arm64 wheel excludes all CUDA tooling. GPU acceleration goes through Apple's Metal/MPS backend instead, which is already part of macOS — so no CUDA runtime needs to be bundled.
>
> **Note on `huggingface_hub`:** The CLI entry point shipped with this package was renamed from `huggingface-cli` to `hf` in recent versions. The login command is now `uv run hf auth login` (see [Section 14](#14-huggingface--wb-login)). If you ever need to check what CLI commands a package registers, inspect `.venv/lib/python3.12/site-packages/<package>.dist-info/entry_points.txt`.

### 8.3. Reproducing the Environment on Another Machine

When you run `uv add`, two files are automatically maintained.

- **`pyproject.toml`** — your declared dependencies (e.g. `torch`, `trl`). The source of truth, human-readable, and editable directly. `uv` always resolves `uv.lock` from this file, never the other way around.
- **`uv.lock`** — the fully resolved dependency tree with exact versions for every package and sub-dependency. Auto-generated by `uv`, never edited manually.

The relationship between them is as follows.

- `uv lock` — update `uv.lock` from `pyproject.toml`.
- `uv sync` — update `uv.lock` from `pyproject.toml` if needed, then update `.venv` from `uv.lock`.
- `uv sync --frozen` — skip updating `uv.lock` entirely, update `.venv` directly from whatever `uv.lock` currently says.
- `uv sync --locked` — error if `uv.lock` is inconsistent with `pyproject.toml`, otherwise update `.venv` from `uv.lock` (useful in CI pipelines to enforce the lock file was committed correctly).

If you edit `pyproject.toml` manually (e.g. to adjust version constraints or add multiple packages at once), run `uv sync` afterward to apply the changes to `uv.lock` and `.venv`.

Commit both files to git:

```bash
git add pyproject.toml uv.lock
git commit -m "Add project dependencies"
```

On another machine, recreating the exact environment is a single command:

```bash
git clone <your-repo>
cd rlhf-course
uv sync
```

`uv sync` reads `uv.lock` and recreates `.venv` with exactly the same versions, with no resolution step needed, which is why it is fast. If the other machine is a different architecture (e.g. Linux x86 instead of Apple Silicon), `uv sync` re-resolves platform-specific wheels automatically while keeping all versions consistent where possible.

If you ever need a traditional `requirements.txt` for compatibility with tools that do not understand `uv.lock`:

```bash
uv pip freeze > requirements.txt
```

For your own projects between machines, `uv.lock` + `uv sync` is the better mechanism, being more precise and fully automated.

### 8.4. Run Commands in the Project Environment

```bash
# Option A: prefix commands with `uv run` (no activation needed — preferred)
uv run python my_script.py
uv run jupyter lab

# Option B: activate the venv traditionally
source /Volumes/ML_Workspace/projects/rlhf-course/.venv/bin/activate
python my_script.py
```

### 8.5. Repeat for Each New Project

```bash
# Example: Kaggle competition
mkdir /Volumes/ML_Workspace/projects/kaggle-titanic
cd /Volumes/ML_Workspace/projects/kaggle-titanic
uv init --python 3.12
uv add pandas scikit-learn xgboost lightgbm matplotlib seaborn jupyterlab
```

Each project gets its own isolated `.venv`. Wheels are cached in `/Volumes/ML_Workspace/cache/uv` and hard-linked, with no redundant downloads across projects.

---

## 9. Jupyter Lab Setup

### 9.1. When to Register a Jupyter Kernel

If you launch Jupyter Lab via `uv run` from inside the project directory, the correct environment is used automatically, with no kernel registration needed:

```bash
cd /Volumes/ML_Workspace/projects/rlhf-course
uv run jupyter lab
```

Register a kernel manually only if you want to run a single Jupyter Lab instance and switch between multiple project environments within it:

```bash
cd /Volumes/ML_Workspace/projects/rlhf-course
source .venv/bin/activate
python -m ipykernel install --user --name rlhf-course --display-name "Python (rlhf-course)"
```

Repeat for each additional project you want accessible as a named kernel.

### 9.2. Switch Kernels in Notebooks

In Jupyter Lab: top-right kernel selector → choose 'Python (rlhf-course)'. Each notebook can use a different project's environment.

---

## 10. Editor: VS Code + Claude Code

### 10.1. Why Not Cursor?

Cursor ($20/month) is a popular AI-native IDE, but given you already have Claude Pro (which includes Claude Code), paying for Cursor is redundant for this use case.

- **Claude Code** — included in your Claude Pro plan, handles deep agentic tasks: scaffolding projects, debugging across multiple files, refactoring training loops, running and iterating autonomously.
- **VS Code** — free, handles daily editing: writing notebooks, browsing code, running scripts.
- **Claude.ai** — theory explanations, paper walkthroughs, architecture discussions.

Cursor's main advantage is fast inline autocomplete while actively typing. For ML work, writing training loops, running experiments, and iterating on RLHF pipelines, you spend more time thinking and running code than typing boilerplate, so that advantage matters less.

### 10.2. Install VS Code

Download from https://code.visualstudio.com and drag to Applications.

Install the `code` CLI so you can control VS Code from the terminal: `Cmd+Shift+P` → 'Shell Command: Install 'code' command in PATH'.

Common uses of the `code` CLI:

```bash
code .                                              # open current directory
code myfile.py                                      # open a specific file
code /Volumes/ML_Workspace/projects/rlhf-course     # open a project
```

> **Where extensions are stored:** VS Code installs extensions to `~/.vscode/extensions/` on your internal SSD, not the external drive. This is intentional: extensions are small (a few MB each) and tied to VS Code itself, not to any specific project. Unlike model weights or package caches, they do not grow large enough to warrant moving to the external NVMe.
>
> **macOS permission prompt:** When running `code --install-extension` for the first time, macOS will ask for permission to access the external drive. This is triggered by VS Code scanning connected volumes to detect project environments, not because extensions are being installed there. Grant the permission; without it VS Code cannot detect your `.venv` folders on the external drive. You will only be asked once.

### 10.3. Install Extensions

```bash
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
code --install-extension ms-python.vscode-pylance
code --install-extension eamodio.gitlens
code --install-extension charliermarsh.ruff
```

What each extension does.

- **`ms-python.python`** — core Python support: syntax highlighting, linting, interpreter selection. How VS Code detects and uses your project's `.venv`.
- **`ms-toolsai.jupyter`** — open, run, and edit `.ipynb` notebook files directly in VS Code without a browser.
- **`ms-python.vscode-pylance`** — fast type checking, intelligent autocomplete, 'go to definition' and 'find references' as you write code.
- **`eamodio.gitlens`** — enhances git support. Most useful feature: inline git blame showing who last changed each line of code and when.
- **`charliermarsh.ruff`** — fast Python linter and formatter written in Rust. Catches style issues and common bugs as you type, auto-formats on save. Replaces `flake8`, `black`, and `isort` in one extension.

> **Note on a `uv` extension:** There is no official `uv` VS Code extension from Astral. The core Python extension (`ms-python.python`) already detects `uv`-managed `.venv` folders automatically, with no additional extension needed.

### 10.4. Auto-Detection of uv Environments

When you open `/Volumes/ML_Workspace/projects/rlhf-course` in VS Code, it automatically detects the `.venv` folder and selects the correct Python interpreter. No manual configuration is needed.

---

## 11. Git Configuration

```bash
git config --global user.name "Your Name"
git config --global user.email "123456789+your-username@users.noreply.github.com"
git config --global init.defaultBranch main
git config --global core.editor "code --wait"
```

> **What `core.editor "code --wait"` does:** This sets VS Code as the editor git opens when it needs you to write something interactively, most commonly when you run `git commit` without the `-m` flag, `git rebase -i`, or `git merge`. Git opens the file in VS Code, then waits for you to write your message and close the tab before continuing. The `--wait` flag is essential: without it, git would open VS Code and immediately proceed without waiting, since VS Code is a GUI app that opens in its own window. Without this setting, git falls back to `vim` or `nano` in the terminal.
>
> **Why the `noreply` email?** GitHub blocks pushes that expose your real email address if the 'Block command line pushes that expose my email' privacy setting is enabled (Settings → Emails). Using your GitHub-provided no-reply address (`123456789+your-username@users.noreply.github.com`) prevents this. Find your exact no-reply address at GitHub → Settings → Emails.

```bash
# SSH key for GitHub — use the same no-reply address as the comment
ssh-keygen -t ed25519 -C "123456789+your-username@users.noreply.github.com" -f ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
# → Add to GitHub: Settings → SSH Keys
```

#### Global `.gitignore` — exclude common generated files from all repos

```bash
cat > ~/.gitignore_global << 'EOF'
.venv/
.vscode/
__pycache__/
*.pyc
.DS_Store
*.egg-info/
dist/
build/
.ipynb_checkpoints/
wandb/
EOF

git config --global core.excludesfile ~/.gitignore_global
```

> **`.vscode/`** — VS Code creates this folder inside project directories to store workspace-specific settings (chosen interpreter path, debug configs, extension recommendations). These are personal editor preferences that should not be committed to git and imposed on collaborators who may use different editors or local paths.

---

## 12. Kaggle CLI

```bash
cd /Volumes/ML_Workspace/projects/kaggle-titanic
uv add kaggle
```

Kaggle authenticates via two environment variables already set in [Section 6](#6-shell-environment-variables):

```bash
export KAGGLE_USERNAME="YOUR_KAGGLE_USERNAME"
export KAGGLE_API_TOKEN="YOUR_API_TOKEN_HERE"
```

To get these values: kaggle.com → Settings → API → Create New API Token. The current Kaggle UI displays your token as a string on screen rather than downloading a file — copy it and paste it as `KAGGLE_API_TOKEN` in your `~/.zshrc`. Your username is your Kaggle account username.

> **Why environment variables instead of `~/.kaggle/kaggle.json`?** Kaggle's current recommended approach is the `KAGGLE_API_TOKEN` environment variable. The older `kaggle.json` file approach still works but is no longer the default flow in the UI. Both methods are equivalent; the environment variable approach is cleaner since credentials stay in `~/.zshrc` alongside all other ML tooling config.
>
> **Persistence:** These variables are in `~/.zshrc`, so they are available in every terminal session automatically, with no need to re-authenticate after reboots.

Verify and download a competition dataset:

```bash
# Verify authentication works
uv run kaggle competitions list

# Download a competition dataset
uv run kaggle competitions download -c titanic -p /Volumes/ML_Workspace/kaggle/titanic
```

---

## 13. Ollama (Local LLM Inference)

Ollama model blobs are large (4–70 GB each), making storing them on the external NVMe essential.

### 13.1. Install Ollama

```bash
brew install ollama
```

After installation, start Ollama as a background service that launches automatically at login:

```bash
brew services start ollama
```

> **Do not run `ollama serve` manually.** Once the background service is running, `ollama serve` will fail with 'address already in use' because the service already owns port 11434. The background service is always running — just use `ollama` commands directly.

### 13.2. Redirect Model Storage to External Drive

Background services launched by `brew services` run in a separate environment from your terminal and do not read `~/.zshrc`. This means the `OLLAMA_MODELS` path set in [Section 6](#6-shell-environment-variables) is invisible to the service — models will default to `~/.ollama/models/` on your internal SSD.

Fix this by creating a launchd plist that sets the required environment variables for the background service:

```bash
# Stop the service first
brew services stop ollama

# Create the launchd plist
cat > ~/Library/LaunchAgents/ollama.environment.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>ollama.environment</string>
    <key>ProgramArguments</key>
    <array>
        <string>sh</string>
        <string>-c</string>
        <string>launchctl setenv OLLAMA_MODELS /Volumes/ML_Workspace/ollama/models &amp;&amp; launchctl setenv OLLAMA_FLASH_ATTENTION 1 &amp;&amp; launchctl setenv OLLAMA_KV_CACHE_TYPE q8_0</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
EOF

# Load the plist
launchctl load ~/Library/LaunchAgents/ollama.environment.plist

# Restart the service
brew services start ollama
```

### 13.3. Pull a Model

```bash
ollama pull llama3.2:3b

# Verify it landed on the external drive
ls /Volumes/ML_Workspace/ollama/models/
```

If you already pulled a model before setting up the plist (it will be in `~/.ollama/models/`), move it rather than re-downloading:

```bash
mv ~/.ollama/models/* /Volumes/ML_Workspace/ollama/models/
```

Verify Ollama can find the model from its new location:

```bash
ollama list
# Should show llama3.2:3b
```

---

## 14. HuggingFace & W&B Login

These are one-time setup steps. Credentials are stored persistently on disk and survive terminal restarts and reboots — you do not need to repeat these logins.

```bash
cd /Volumes/ML_Workspace/projects/rlhf-course

# HuggingFace (needed for gated models: Llama 3, Mistral, Gemma)
uv run hf auth login
# Paste token from huggingface.co → Settings → Access Tokens
# When prompted "Add token as git credential?" → choose Yes
# This stores the token in macOS Keychain so git operations
# against private HuggingFace repos work without re-authenticating.
```

> **Note on the CLI name:** The HuggingFace CLI entry point was renamed from `huggingface-cli` to `hf` in recent versions of `huggingface_hub`. The login subcommand moved under the `auth` group: `hf auth login`. If you ever see `huggingface-cli` in older guides or documentation, substitute `hf auth` accordingly.
>
> **Where tokens are stored:** Both of the following paths are on your external NVMe because of the `HF_HOME` environment variable set in [Section 6](#6-shell-environment-variables). If you cannot remember the paths, `echo $HF_HOME` points you to the right directory.
> - `$HF_HOME/token` — active token used by the HuggingFace library.
> - `$HF_HOME/stored_tokens` — all named tokens you have logged in with.
> - macOS Keychain (`osxkeychain`) — used for git credential authentication.

```bash
# Weights & Biases (experiment tracking)
uv run wandb login
# Paste API key from wandb.ai → Settings
```

> **Where the W&B token is stored:** `~/.netrc` on your internal SSD. W&B appends an entry in the format:
> ```
> machine api.wandb.ai
>   login user
>   password YOUR_API_KEY
> ```
> `~/.netrc` is a standard Unix credentials file used by many CLI tools. Its permissions are `600` (owner read/write only), the same as any credentials file.

---

## 15. Smoke Test

```bash
cd /Volumes/ML_Workspace/projects/rlhf-course
```

Create `smoke_test.py`:

```python
import os
import torch
import mlx.core as mx
import gymnasium as gym
from transformers import AutoTokenizer
from trl import DPOTrainer

print("=== Environment Check ===")
print(f"PyTorch:        {torch.__version__}")
print(f"MPS available:  {torch.backends.mps.is_available()}")
print(f"MPS built:      {torch.backends.mps.is_built()}")
print(f"MLX device:     {mx.default_device()}")

print("\n=== Cache Paths (all should be on /Volumes/ML_Workspace) ===")
print(f"HF_HOME:        {os.environ.get('HF_HOME')}")
print(f"TORCH_HOME:     {os.environ.get('TORCH_HOME')}")
print(f"UV_CACHE_DIR:   {os.environ.get('UV_CACHE_DIR')}")

print("\n=== Gymnasium ===")
env = gym.make("CartPole-v1")
obs, _ = env.reset()
print(f"CartPole obs shape: {obs.shape}")
env.close()

print("\n✅ All checks passed.")
```

> **Note on the `trl` import:** `PPOConfig` was removed in recent versions of TRL. The smoke test uses `DPOTrainer` instead, which verifies TRL is correctly installed and importable. To see the full list of available classes in your installed version, run `uv run python -c "import trl; print(dir(trl))"`.

```bash
uv run python smoke_test.py
```

---

## 16. Daily Workflow Cheatsheet

### 16.1. Start Working on a Project

```bash
cd /Volumes/ML_Workspace/projects/rlhf-course
uv run jupyter lab          # no activation needed
# or
source .venv/bin/activate   # traditional activation
```

### 16.2. Open Project in VS Code

```bash
code /Volumes/ML_Workspace/projects/rlhf-course
# or from inside the project directory:
cd /Volumes/ML_Workspace/projects/rlhf-course
code .
```

### 16.3. Add a New Package

```bash
uv add stable-baselines3          # adds to pyproject.toml + installs
uv add --dev pytest               # dev-only dependency
```

### 16.4. Remove a Package

```bash
uv remove stable-baselines3
```

### 16.5. Reproduce an Environment on Another Machine

```bash
# uv.lock is auto-generated — commit it to git alongside pyproject.toml
# On another machine:
git clone <your-repo>
cd rlhf-course
uv sync                           # recreates .venv from uv.lock exactly
```

### 16.6. Update Packages

```bash
uv pip list --outdated
uv lock --upgrade-package torch   # upgrade one package
uv lock --upgrade                 # upgrade all
uv sync                           # apply updated lock to venv
```

### 16.7. Monitor System Resources During Training

```bash
htop                                                         # CPU and RAM
sudo powermetrics --samplers gpu_power,cpu_power -i 1000    # GPU + Neural Engine power
df -h /Volumes/ML_Workspace                                  # disk usage
du -sh /Volumes/ML_Workspace/cache/uv                       # uv cache size
```

---

## 17. Maintenance Reference

| Task | Command |
|---|---|
| Install new Python version | `uv python install 3.13` |
| List installed Pythons | `uv python list` |
| Create new project | `mkdir /Volumes/ML_Workspace/projects/X && cd $_ && uv init` |
| Add package | `uv add <package>` |
| Remove package | `uv remove <package>` |
| Sync venv to lock file | `uv sync` |
| Clear uv cache | `uv cache clean` |
| Check cache size | `du -sh /Volumes/ML_Workspace/cache/uv` |
| List project dependencies | `uv pip list` |
| Export requirements.txt | `uv pip freeze > requirements.txt` |
| Stop Ollama service | `brew services stop ollama` |
| Start Ollama service | `brew services start ollama` |
| List Ollama models | `ollama list` |

---

## 18. `uv` vs Miniforge — Why `uv`

| | Miniforge (conda) | uv |
|---|---|---|
| Install speed | Slow (minutes for large envs) | Very fast (seconds) |
| Python management | Separate (conda itself) | Built-in |
| Package source | conda-forge + PyPI | PyPI only |
| Env isolation | Named global envs | Per-project `.venv` |
| Dependency locking | `environment.yml` (loose) | `uv.lock` (exact, reproducible) |
| Disk efficiency | Each env copies packages | Shared cache + hard links |
| Apple Silicon ML packages | ✅ All available | ✅ All available on PyPI |
| Required for this stack | No | No — equal capability |

conda's package channels offer no advantage for this stack — every library (e.g. PyTorch MPS, MLX, TRL, PEFT, Gymnasium) ships native Apple Silicon wheels on PyPI. `uv` is faster, simpler, and produces more reproducible environments.
