# Research relay for Claude Code and Codex

This local controller transfers research work between Claude Code and Codex.
It starts with Claude. When Claude reaches a usage limit, Codex continues the work.
The controller returns to Claude after the reported reset time and a 30-second margin.
If both providers reach a limit, the controller waits.

The controller uses Python 3.11 or later and the installed provider CLIs.
It needs no additional Python package. Run it on Linux or WSL from the repository root.
Keep the computer awake during the run.

## Create a run

1. Write the research goal in a text file. Include the expected output, required checks, and stopping conditions.
2. Sign in with the existing `claude` and `codex` CLIs.
3. Create a run. Replace `/tmp/research-goal.md` with your goal file.

```bash
.venv/bin/python -m tools.agent_relay init \
  --workspace . \
  --goal-file /tmp/research-goal.md \
  --run-dir artifacts/agent_relay/night
```

4. Check the local commands and workspace lock.

```bash
.venv/bin/python -m tools.agent_relay doctor --run-dir artifacts/agent_relay/night
```

The `init` and `doctor` commands send no model requests.
The command does not check account access, model access, or available quota.
The controller uses your existing CLI authentication and model settings.
Existing API credentials or extra-usage settings can affect billing.
The controller does not purchase credits or enable extra usage.

## Start a run

Start an eight-hour run in the current terminal:

```bash
.venv/bin/python -m tools.agent_relay run \
  --run-dir artifacts/agent_relay/night --hours 8
```

To keep the controller active after you close the terminal, start it in `tmux`:

```bash
tmux new-session -s research-relay \
  '.venv/bin/python -m tools.agent_relay run --run-dir artifacts/agent_relay/night --hours 8'
```

Use `Ctrl-b`, then `d`, to detach from `tmux`.
Use `tmux attach-session -t research-relay` to return.
The controller can use `tmux`. Its provider processes must not create detached sessions.

Check or stop the run from another terminal:

```bash
.venv/bin/python -m tools.agent_relay status --run-dir artifacts/agent_relay/night
.venv/bin/python -m tools.agent_relay stop --run-dir artifacts/agent_relay/night
```

`Ctrl-c` and `SIGTERM` also request a stop.
After you correct a blocked task or stop a run, resume it:

```bash
.venv/bin/python -m tools.agent_relay resume \
  --run-dir artifacts/agent_relay/night --hours 8
```

Each `run` or `resume` command sets a new duration limit.
A completed goal requires a new run directory.
The controller returns exit code 2 for a blocked run or a command error.
It returns exit code 0 for a complete or paused run. Check `state.json` to distinguish these states.

## Shared state

| File | Purpose |
| --- | --- |
| `GOAL.md` | The original research goal. |
| `TASKS.md` | Tasks, assignments, and task status. |
| `FINDINGS.md` | Results, sources, checks, and unresolved questions. |
| `HANDOFF.md` | Current progress and the next useful actions. |
| `checkpoints/*.json` | Structured results from completed work periods. |
| `logs/*.jsonl` | Provider events, including visible messages and tool activity. |
| `logs/*.stderr` | Provider diagnostics. |
| `events.jsonl` | Controller transitions and errors. |
| `state.json` | Provider session IDs, cooldowns, and run status. |
| `config.json` | Workspace, commands, time limits, and team setting. |

Each provider resumes its own saved session.
On every transfer, the incoming provider reads the shared files and newest checkpoint.
The controller does not convert private model state or native team state between providers.
Agents must save findings after each useful step. A hard limit can prevent a final checkpoint.
In that case, the next provider uses the last saved files and the event logs.
The controller accepts completion only from a valid result for the current work period.
The agent must cite evidence for completion. The controller cannot independently prove arbitrary research claims.

Run directories have mode `0700`. They can contain private research and provider output.
The repository ignores the default `artifacts/` location. Each run also contains its own `.gitignore`.

## Timing and failure handling

The default work period is ten minutes. Use `init --slice-minutes 5` to select five minutes.
Agents receive the deadline in their prompt. Codex also receives a request to save progress before the deadline.
At the deadline, the controller interrupts the active provider and stops its process group.
It starts the next provider only after cleanup. Cleanup can add up to about 36 seconds.

Claude teams use the in-process mode. The incoming Claude session recreates teammates when necessary.
Use `init --no-claude-teams` to disable the team request.
Detached processes and remote jobs are outside the process group. The agent instructions prohibit starting them.
Existing jobs outside the controller remain outside its control.

Claude reset times come from `rate_limit_event.rate_limit_info.resetsAt`.
Codex reset times come from `account/rateLimits/read` and its update events.
The controller uses the latest reset among exhausted primary and secondary Codex windows.
At the reset time, it retries the provider. A new rejection can extend the cooldown.
If no usable reset time is available, it retries after 15 minutes plus the reset margin.
Use `init --retry-minutes 30` to change that fallback.
The controller does not guess reset times from localized display text.

Ordinary failures have a separate retry count. After three failures, that provider stops receiving work.
If both providers fail repeatedly, the run becomes blocked.
A human-input request also blocks the run.
Codex terminal permission or policy denials also block the run, including `tooManyDenials`,
`cyberPolicy`, and `misalignmentPolicyViolation`. The controller preserves the error details.
It does not retry the denied task or transfer it to another provider.
Claude can complete a task through permitted actions after an earlier permission denial.
Those earlier denials remain in the provider log. They do not override a valid final checkpoint.
A blocked checkpoint, or a denial without a valid checkpoint, blocks the run.
An error result with permission denials also blocks the run.
Read the logs before you use `resume`. Resume clears failure counts but preserves quota cooldowns.

Claude uses `--permission-mode auto` and `--permission-prompts none`.
Codex uses the `workspace-write` sandbox and `auto_review` approval reviewer.
The controller does not approve requests that require a person.
Your account and model must support the selected approval mode.
The controller preserves CLI settings, project instructions, hooks, and permission rules.

Only one controller can hold a workspace lock.
Provider processes inherit the lock so a controller crash cannot immediately allow a competing run.
After `SIGKILL` or a host crash, inspect the recorded process ID before restarting.
Do not remove the lock file to force another controller to start.
The `stop` file requires a running controller to act on it.

## Design and validation

The implementation separates persistent state, provider protocols, process control, and CLI commands.
It uses atomic state replacement, JSON event streams, and process groups.
No shell evaluates provider command arguments.

The implementation sequence was:

1. Define tests for limits, state persistence, and provider selection.
2. Implement the store and provider protocols.
3. Implement process cleanup and the scheduler.
4. Test the complete flow with local provider substitutes.
5. Run an independent review and correct valid findings.

Run the focused tests:

```bash
.venv/bin/python -m pytest tests/test_agent_relay.py -q \
  --cov=tools.agent_relay --cov-report=term-missing --cov-fail-under=80
```

These tests send no model requests. They check protocol messages, transfer timing, restart state, failure handling, and process cleanup.
The interface checks used Claude Code 2.1.301 and codex-cli 0.163.0.
Live checks on 2026-10-16 confirmed a complete Codex run and a transfer from Claude to Codex and back to Claude.
The transfer test used real provider processes and a simulated Claude usage limit.
Codex stopped at the reset deadline. Claude resumed its saved session and checked the file from Codex.
None of the observed Codex processes remained after the transfer.
These checks did not test actual quota exhaustion, active teammates, or an overnight research run.

Protocol references: [Claude programmatic mode](https://code.claude.com/docs/en/headless),
[Claude team limitations](https://code.claude.com/docs/en/agent-teams#limitations),
and [Codex App Server](https://learn.chatgpt.com/docs/app-server).
