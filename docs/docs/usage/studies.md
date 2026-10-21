---
sidebar_position: 3.55
title: Studies
description: Follow durable studies in the Studies panel, read rankings with ties, and set the server-enforced budgets that halt a study before it runs too long.
---

# Studies

A **study** runs a set of graph variants a fixed number of times each, with a different seed per repetition, and ranks the variants on one exact objective. The server owns the study from start to finish: it keeps running after the browser tab that submitted it closes, survives a server restart, and resumes without running a completed repetition again. How to submit one, the ranking rule and the restart policy are in [Run Queue, Studies](./run-queue#studies); the routes are in the [API Reference](../advanced/api-reference#runs-and-sweeps).

This page covers the **Studies panel** in the web app, the **budgets** that stop a study before it uses more than you allowed, and studies that a **plugin** submits.

## Opening the Studies panel

Open the **Runs** tab at the bottom of the editor and click **Studies** in its toolbar. The panel lists every study on the server, newest first, whichever browser, script or plugin submitted it. Nothing ties a study to the tab that started it: open the panel in another browser, or after closing the original tab, and the same studies are there. The panel reads the server again every 2 seconds while it is open.

Each row shows the study's name, its status, who submitted it (**API** for `POST /api/studies`, or the plugin's id), its progress as completed repetitions out of scheduled ones, and when it was created. A repetition counts as completed once its current attempt succeeded, failed or timed out. Click a name to open the study. **Escape** goes back from a study to the list, and from the list closes the panel.

| Status | Meaning |
| --- | --- |
| Queued, Running | The study is scheduling its children. |
| Cancelling | A cancel was sent; the study waits until every child in flight has stopped. |
| Finished | Every scheduled repetition ended (`succeeded`). Finished is about the schedule: a repetition that failed is a result. |
| Failed | The study itself could not continue, for example the run service refused a child. |
| Cancelled | Stopped by a cancel. Completed repetitions are kept. |
| Interrupted | The server restarted, or a child run was stopped outside the study. It waits for **Resume**. |
| Halted | A budget ran out. See [Budgets](#budgets). |

## Following a study

An open study shows, from top to bottom:

- **The objective** (series, producer node and direction), the number of variants and repetitions, and the study seed. A study a plugin submitted names the plugin.
- **A notice** when the study is halted (which budget, its limit and what was used) or interrupted (it waits for a resume).
- **Cancel study** and **Resume**, when the status allows them. A halted study resumes through the budget form instead (see [Resuming a halted study](#resuming-a-halted-study)).
- **Budget**: runs submitted, children in flight, the summed active time and the per-run limit, each against its limit, and **used up** next to a budget that ran out.
- **Rankings**: the ranked variants best first, with mean, sample standard deviation and count. A variant is ranked only when every scheduled repetition succeeded and recorded the objective. Variants with exactly equal means share a rank and read **Tie with** the other variants; the next rank skips (1, 1, 3). No run-time tie-break is applied.
- **Repetitions**: one row per variant and one column per repetition. Each cell shows the current attempt's status, its objective value, **attempt n** when the repetition was retried, and the first 8 characters of its run id; click the id to open that run in the Runs panel. The row ends with the variant's mean, standard deviation, count and whether it is ranked, or why not (for example `r2 Timed out`).
- **Event log**: the study's audit log, newest first: every submission, start, finish, stop request, cancel, resume, halt and interruption, with its cursor and time.

## Budgets

A study's `budget` is enforced by the server, whether or not anyone is watching. Every key is optional except that `max_parallel_runs` defaults to 1.

| Key | Limit | When it is reached |
| --- | --- | --- |
| `max_runs` | Attempts the study may submit, retries and lost submissions included. At least `variants x repetitions`. | No further child is submitted. Once the children in flight end, the study is **halted**. |
| `max_parallel_runs` | Children queued or running at once (1 to 32). | The next child waits for one to end. |
| `max_run_seconds` | Execution time of one child, from its start (queue time does not count). | The server cancels that child. Its repetition ends **timed out**: a result, like a failure, so it is not run again and its variant is not ranked. The study continues. |
| `max_active_seconds` | Summed execution time of every child, retries included. | The server cancels the children in flight (their repetitions read **cancelled**) and the study is **halted** once they have stopped. |

The time limits are seconds, above 0 and up to 30 days. Before stopping a child for a budget, the server records the budget on the attempt (`stop_reason`), so a child the study stopped is never mistaken for one stopped from outside, which would interrupt the study. Stopping is the run queue's cooperative cancel: a child on this server stops at its next check (training, evaluation and sampling nodes check every batch), and a child on a [worker](./remote-workers) is terminated by its worker when it has not stopped within the grace period. Active time can therefore end slightly above `max_active_seconds`.

A **halted** study keeps every result it has: completed repetitions, their values, the rankings so far and the event log. `GET /api/studies/{id}` returns `halt` (`budget`, `limit`, `used`, `unfinished_repetitions` and `at`) and `budget_usage` (what each budget has used and which one ran out), and the event log ends with `study_halted`, whose payload carries the partial ranking. A halted study is not active: it does not appear under `?status=queued|running|cancel_requested`, it survives a restart unchanged, and nothing runs until it is resumed or cancelled.

### Resuming a halted study

A halted study shows two fields, **Max runs** and **Max active seconds**, filled with the current limits. Raise the one that ran out (an empty field means no limit) and click **Resume with this budget**. The study continues with the new budget: repetitions whose child the budget stopped get a new attempt with the same seed, and completed repetitions are not run again. Through the API, send the whole new budget:

```bash
curl -X POST http://127.0.0.1:8000/api/studies/$STUDY/resume \
  -H "X-CodefyUI-Token: $TOKEN" -H "Content-Type: application/json" \
  -d '{"budget": {"max_parallel_runs": 1, "max_active_seconds": 3600}}'
```

A resume whose budget is still used up is refused with `409` and a message that names the budget, since the study would halt again at once. `budget` can be sent to resume a study in any resumable status; keys left out of it take their defaults.

## Studies from plugins

A plugin can hand a study to the server and stop holding it in its page. Graph Copilot's experiments are the intended user: once you approve a plan in its chat, the plugin can submit the study as a job of kind `study` to `POST /api/plugins/graph-copilot/jobs` and follow it from any browser. The job is an ordinary study whose `owner` is the plugin: it appears in this panel with the plugin's id under **Submitted by**, it can be cancelled and resumed here, and its budgets apply as above. The plugin sees only its own jobs through its routes. Delegating Copilot's execution to this API is a change in the plugin's own repository; the routes are listed in the [API Reference](../advanced/api-reference#plugin-jobs).

## See also

- [Run Queue, Studies](./run-queue#studies): submitting a study, seeds, idempotency, the ranking rule and restarts
- [Remote Workers](./remote-workers) and [Multi-GPU Execution](./multi-gpu): running a study's children on workers with `options.target`
- [API Reference](../advanced/api-reference#runs-and-sweeps): the `/api/studies` routes, and [plugin jobs](../advanced/api-reference#plugin-jobs)
