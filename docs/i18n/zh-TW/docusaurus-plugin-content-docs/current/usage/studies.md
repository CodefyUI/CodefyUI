---
sidebar_position: 3.55
title: 實驗研究（Studies）
description: 在 Studies 面板追蹤持久化的 study、閱讀含平手標示的排名，並設定由伺服器強制執行的預算，讓 study 在超出額度前暫停。
---

# 實驗研究（Studies）

**study** 會把一組圖的變體各執行固定次數，每次重複使用不同的種子，並依單一明確的目標替變體排名。study 從頭到尾由伺服器負責：送出它的瀏覽器分頁關閉後仍會繼續執行，伺服器重新啟動後依然存在，繼續執行時也不會重跑已完成的重複。如何送出 study、排名規則與重新啟動的處理方式，請見[執行佇列：實驗研究](./run-queue#studies)；各 route 列在 [API 參考](../advanced/api-reference#runs-and-sweeps)。

本頁說明網頁介面中的 **Studies 面板**、在 study 用超過你允許的額度之前讓它停下的**預算**，以及由**外掛**送出的 study。

## 開啟 Studies 面板 {/* #opening-the-studies-panel */}

開啟編輯器下方的**執行任務**分頁，按工具列上的 **Studies**。面板會由新到舊列出伺服器上的每一個 study，不論它是由哪個瀏覽器、腳本或外掛送出。study 與啟動它的分頁沒有綁定關係：在另一個瀏覽器開啟面板，或關掉原本的分頁之後再開啟，看到的都是同一批 study。面板開啟期間每 2 秒重新讀取一次伺服器。

每一列顯示 study 的名稱、狀態、送出者（`POST /api/studies` 送出的顯示 **API**，否則顯示外掛的 id）、以「已完成的重複 / 排定的重複」表示的進度，以及建立時間。一次重複的目前 attempt 成功、失敗或逾時，就算完成。按名稱即可開啟 study。按 **Escape** 會從 study 回到清單，在清單時則關閉面板。

| 狀態 | 意義 |
| --- | --- |
| 排隊中、執行中 | study 正在排程它的 child。 |
| 取消中 | 已送出取消；study 會等到每個執行中的 child 都停下。 |
| 已完成 | 每一次排定的重複都已結束（`succeeded`）。「已完成」指的是排程：失敗的重複也是一個結果。 |
| 失敗 | study 本身無法繼續，例如 run service 拒絕了某個 child。 |
| 已取消 | 因取消而停止。已完成的重複會保留。 |
| 已中斷 | 伺服器曾重新啟動，或有 child run 在 study 之外被停止。它會等待**繼續**。 |
| 已暫停（預算） | 某項預算用完了。請見[預算](#budgets)。 |

## 追蹤 study {/* #following-a-study */}

開啟的 study 由上而下顯示：

- **目標**（series、產生它的節點與方向）、變體與重複的數量，以及 study 種子。由外掛送出的 study 會標出外掛名稱。
- study 暫停時（哪一項預算、上限與用量）或中斷時（等待繼續）顯示的**提示**。
- 狀態允許時顯示**取消 study** 與**繼續**。暫停的 study 改由預算表單繼續（請見[繼續暫停的 study](#resuming-a-halted-study)）。
- **預算**：已送出的 run、執行中的 child、累計執行時間與單一 run 上限，各自對照其上限；用完的預算旁會標示**已用完**。
- **排名**：已排名的變體由最佳開始列出，包含平均、樣本標準差與次數。只有每一次排定的重複都成功並記錄到目標值時，變體才會排名。平均值完全相同的變體共用同一個名次，並顯示**與……平手**；下一個名次則跳號（1、1、3）。執行時不做任何平手裁定。
- **重複**：每個變體一列，每次重複一欄。每一格顯示目前 attempt 的狀態、目標值、重試過的重複會顯示**第 n 次嘗試**，以及 run id 的前 8 個字元；按 id 即可在執行任務面板開啟該 run。每列最後是該變體的平均、標準差、次數，以及是否排名或未排名的原因（例如 `r2 逾時`）。
- **事件紀錄**：study 的稽核紀錄，由新到舊：每一次送出、開始、結束、停止要求、取消、繼續、暫停與中斷，以及它的 cursor 與時間。

## 預算 {/* #budgets */}

study 的 `budget` 由伺服器強制執行，不論是否有人在看。每個欄位都是選填，`max_parallel_runs` 預設為 1。

| 欄位 | 上限 | 達到上限時 |
| --- | --- | --- |
| `max_runs` | study 可送出的 attempt 數，包含重試與遺失的送出。至少為 `variants x repetitions`。 | 不再送出任何 child。執行中的 child 結束後，study 會**暫停**（`halted`）。 |
| `max_parallel_runs` | 同時排隊或執行中的 child 數（1 到 32）。 | 下一個 child 等到有 child 結束。 |
| `max_run_seconds` | 單一 child 的執行時間，從它開始執行起算（排隊時間不算）。 | 伺服器取消該 child。該次重複以**逾時**（`timed_out`）結束：這是一個結果，與失敗相同，不會再執行，所屬變體也不會排名。study 會繼續。 |
| `max_active_seconds` | 所有 child 的累計執行時間，包含重試。 | 伺服器取消執行中的 child（這些重複顯示**已取消**），等它們停下後 study 會**暫停**。 |

時間上限以秒為單位，必須大於 0，最多 30 天。伺服器因預算停止 child 之前，會先把預算記錄在 attempt 上（`stop_reason`），因此 study 自己停止的 child，不會被誤認為是在 study 之外被停止（那會讓 study 中斷）。停止使用的是執行佇列的協作式取消：在這台伺服器上的 child 會在下一次檢查時停止（訓練、評估與取樣節點每個 batch 檢查一次），在 [worker](./remote-workers) 上的 child 若未在寬限期內停止，會由 worker 終止。因此累計執行時間可能略超過 `max_active_seconds`。被停止的 child 仍在執行時，伺服器每 10 秒會再送出一次停止要求，因此因錯誤或伺服器重新啟動而遺失的取消不會就此沒有送出。

**暫停**的 study 會保留它所有的結果：已完成的重複、它們的值、目前的排名與事件紀錄。`GET /api/studies/{id}` 會回傳 `halt`（`budget`、`limit`、`used`、`unfinished_repetitions` 與 `at`）與 `budget_usage`（每項預算的用量，以及哪一項已用完），事件紀錄最後一筆是 `study_halted`，其 payload 帶有部分排名。暫停的 study 不算進行中：它不會出現在 `?status=queued|running|cancel_requested` 的結果中，伺服器重新啟動後維持原狀，在繼續或取消之前不會執行任何東西。指定 `"resume_policy": "auto"` 的 study 若在預算用完後被重新啟動中斷，會變成暫停，而不會自行繼續。

### 繼續暫停的 study {/* #resuming-a-halted-study */}

暫停的 study 會顯示兩個欄位：**run 數上限**與**執行時間上限（秒）**，預先填入目前的上限。提高已用完的那一項（留白表示不限），再按**以此預算繼續**。study 會以新的預算繼續：被預算停止的重複會得到使用相同種子的新 attempt，已完成的重複不會重跑。透過 API 時，請送出完整的新預算：

```bash
curl -X POST http://127.0.0.1:8000/api/studies/$STUDY/resume \
  -H "X-CodefyUI-Token: $TOKEN" -H "Content-Type: application/json" \
  -d '{"budget": {"max_parallel_runs": 1, "max_active_seconds": 3600}}'
```

若繼續時的預算仍是用完的狀態，請求會以 `409` 被拒絕，訊息會指出是哪一項預算，因為 study 會立刻再次暫停。任何可繼續狀態的 study 都可以在繼續時附上 `budget`；未提供的欄位使用預設值。

## 由外掛送出的 study {/* #studies-from-plugins */}

外掛可以把 study 交給伺服器，不必再由自己的頁面保管。Graph Copilot 的實驗正是預期的使用情境：你在它的對話中核准計畫後，外掛可以把 study 以 `study` 類型的 job 送到 `POST /api/plugins/graph-copilot/jobs`，並從任何瀏覽器追蹤它。這個 job 就是一般的 study，其 `owner` 為該外掛：它會出現在這個面板中，**送出者**欄顯示外掛的 id，可以在這裡取消與繼續，預算也同樣適用。外掛透過自己的 route 只看得到自己的 job。讓 Copilot 改用這個 API 執行，是外掛自己儲存庫中的變更；各 route 列在 [API 參考](../advanced/api-reference#plugin-jobs)。

## 延伸閱讀 {/* #see-also */}

- [執行佇列：實驗研究](./run-queue#studies)：送出 study、種子、冪等性、排名規則與重新啟動
- [遠端 Worker](./remote-workers) 與[多 GPU 執行](./multi-gpu)：以 `options.target` 在 worker 上執行 study 的 child
- [API 參考](../advanced/api-reference#runs-and-sweeps)：`/api/studies` 各 route，以及[外掛 job](../advanced/api-reference#plugin-jobs)
