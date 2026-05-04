# Claude Code Terminal — Quick Reference

> Created on: 4 May 2026
>
> Updated on: 4 May 2026

## 1. Starting a Session

Launch Claude Code by running `claude` from within your project directory. The session is scoped to the directory from which it is launched, so always `cd` into the project first.

The table below lists the available startup options.

| Command | Description |
|---|---|
| `claude` | Start a new session in the current directory. |
| `claude "explain this project"` | Start a new session with an opening prompt. |
| `claude --model opus` | Start with a specific model (`opus`, `sonnet`, or `haiku`). |
| `claude --model opusplan` | Hybrid mode, i.e. Opus plans and Sonnet executes. |
| `claude -n "my-session-name"` | Start and name the session for easier retrieval. |

## 2. Resuming a Previous Session

| Command | Description |
|---|---|
| `claude -c` / `claude --continue` | Resume the most recent session in the current directory. |
| `claude -r` / `claude --resume` | Open an interactive picker to browse and select a session. |
| `claude --from-pr <number>` | Resume the session linked to a specific pull request. |
| `/resume` | Switch sessions from inside a running session. |

Session data is stored at two locations.

- `~/.claude/projects/<project>/`, containing per-project session transcripts.
- `~/.claude/history.jsonl`, a global log of every prompt sent across all projects.

## 3. Managing Context Within a Session

### 3.1. Commands

| Command | Description |
|---|---|
| `/context` | Display a live breakdown of token usage by category. |
| `/cost` | Show token consumption and cost for the current session. |
| `/compact` | Summarise conversation history to free up context. |
| `/compact <instructions>` | Steer the summary, e.g. `/compact focus on auth, drop test debugging`. |
| `/clear` | Hard reset, i.e. wipe conversation history (CLAUDE.md still loads). |
| `/rewind` or double-tap `Esc` | Jump back to a previous message and re-prompt from there. |
| `/memory` | View and edit memory files for the current project. |

### 3.2. When to Use Each Command

The three main context-management commands serve distinct purposes.

- `/compact` — use when finishing a subtask but continuing the session; retains a summary of prior turns.
- `/clear` — use when switching to a completely unrelated task; provides a full clean slate.
- `/rewind` — use when Claude has taken the wrong approach; reverts to a prior message without carrying forward the bad turns.

## 4. Switching Models

| Command | Description |
|---|---|
| `/model` | Open the interactive model picker. |
| `/model sonnet` | Switch to Sonnet, the default on most plans. |
| `/model opus` | Switch to Opus, best for complex reasoning tasks. |
| `/model haiku` | Switch to Haiku, optimised for fast and simple tasks. |
| `/model opusplan` | Hybrid mode, i.e. Opus for planning and Sonnet for execution. |
| `/effort high` | Set reasoning depth (`low`, `medium`, `high`, or `xhigh`). |

To set a permanent default model, add the following export to `~/.zshrc` or `~/.bashrc`.

```bash
export ANTHROPIC_MODEL="claude-sonnet-4-6"
```

## 5. Persistent Memory and Project Context

Claude Code supports three mechanisms for persisting instructions and preferences across sessions.

| Mechanism | Description | Location |
|---|---|---|
| `CLAUDE.md` (project) | Instructions written by the user; loaded every session. | `./CLAUDE.md` in the repository root. |
| `CLAUDE.md` (global) | Cross-project preferences written by the user. | `~/.claude/CLAUDE.md`. |
| Auto memory | Notes written by Claude automatically (on by default). | `~/.claude/projects/<project>/memory/`. |

Auto memory captures build commands, debugging fixes, code style preferences, and workflow habits. The files can be reviewed or edited directly. To instruct Claude to remember something explicitly, use natural language, e.g.

```
remember that we use pnpm, not npm
```

## 6. General Slash Commands

| Command | Description |
|---|---|
| `/status` | Show the current model, context usage, and session info. |
| `/rename` | Rename the current session. |
| `/model` | Change the model mid-session. |
| `/help` | List all available slash commands. |
| `/bug` | Report a bug directly to Anthropic. |

## 7. Keyboard Shortcuts

| Shortcut | Description |
|---|---|
| `Ctrl+C` | Cancel the current generation. |
| `Ctrl+R` | Search prompt history, analogous to shell history search. |
| `Esc` `Esc` | Rewind to the previous message. |
| `Shift+Tab` | Toggle plan mode, i.e. Claude reads and plans but does not edit. |

## 8. Tips for Token Efficiency

Managing context window usage reduces cost and keeps responses focused.

- Be explicit about scope, i.e. name the specific file rather than asking Claude to 'look at the codebase'.
- Use subagents for isolation, e.g. 'spin off a subagent to summarise this module', to keep heavy reads out of the main context.
- Run `/clear` between unrelated tasks, treating each as a new session.
- Monitor the status bar, where token percentage is displayed; aim to run `/compact` before reaching 80%.
- Name sessions with `-n` to make them easier to locate later using `claude -r`.
