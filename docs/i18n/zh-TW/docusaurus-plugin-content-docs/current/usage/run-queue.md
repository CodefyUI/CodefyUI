---
sidebar_position: 3.5
title: 執行佇列
description: 依裝置排隊並在背景執行圖形 — 通道、FIFO 順序、並行上限、取消排隊中的 run，以及 cdui run 指令。
---

# 執行佇列

一個 run 由**伺服器**持有，而不是由送出它的客戶端持有。即使關閉分頁、終端機或登出，run 仍會繼續執行；之後可以透過 API 或 CLI 查看。

這就是佇列的用途。一次送出五個訓練工作後關上筆電，伺服器會在 GPU 上逐一執行，而不是五個同時啟動、四十分鐘後耗盡 VRAM。

## 每個裝置一條佇列

每個 run 都依它**解析後的裝置**排程 — `cpu`、`cuda:0`、`mps`。這個字串就是佇列 key，因此每個裝置都有獨立的佇列與並行上限。

| 佇列 key | 同時執行數（預設） | 為什麼 |
| --- | --- | --- |
| `cuda:0`、`cuda:1`、`mps`，任何加速器 | 1 | 同一張卡上的第二個 run 會與第一個競爭同一份 VRAM。若讓兩者同時執行，長時間工作通常會在途中發生 CUDA out-of-memory 錯誤。 |
| `cpu` | 2 | CPU 上的問題是資源競爭，而不是整個 run 失敗；有多餘核心的機器應該用上這些核心。 |

GPU 佇列達到上限時不會延遲 CPU run，兩張不同的卡也不會互相等待。同一條佇列內的 run 依**送出順序**啟動，先送出的先執行。

:::note 同一張卡的別名會併成同一條佇列
`cuda` 與 `cuda:0` 指的是同一個實體裝置，因此會正規化成同一個佇列 key（`cuda:0`），而不是在同一張卡上開兩條各自獨立的 FIFO。`mps` 與 `mps:0` 也一樣。在多張卡的機器上，`cuda` 會跟隨該行程目前的裝置。
:::

:::note `--device auto` 目前會解析成 CPU
裝置解析會把未知或 `auto` 的請求對應到 `cpu`，所以 `auto` 現在是排在 CPU 那條線上。要用顯示卡請明確指定 `cuda`（或 `cuda:0`）。
:::

這項限制控制的是**每個裝置的 run 數量**，與 `CODEFYUI_MAX_PARALLEL_NODES` 不同；後者限制*單一 run 內*可同時執行的節點數。兩項並行上限的效果會相乘。

## 通道（lane）

一個 run 的**通道**記錄它從哪裡來，並決定它怎麼被排程。

| 通道 | 由誰送出 | 排程方式 | 上限 |
| --- | --- | --- | --- |
| `queued`（預設） | `POST /api/runs`、`cdui run` | 加入該裝置的 FIFO | 依裝置，見上表 |
| `interactive` | 畫布（在圖上按 **執行**） | 跳過 FIFO，立即開始 | 同時 2 個，且每個編輯器 session 只能有 1 個 |

畫布會刻意跳過佇列：課堂示範不應排在六小時的訓練工作後面。代價是 interactive run 可能讓同一張卡上的執行數超過該裝置 queued 通道的上限。FIFO 的用途是讓無人看顧的工作依序進行，而不是讓裝置只供一個 run 使用。

### interactive 通道的兩道上限

因為它跳過佇列，interactive 通道改用另外兩種方式設限：

- **最多兩個 interactive run 同時執行**（`CODEFYUI_RUN_INTERACTIVE_MAX_CONCURRENT`），避免大量開啟的分頁耗盡 GPU 資源。
- **每個編輯器 session 只能有一個 run。** 每個開啟的畫布連線會讓由它啟動的 run 使用該 session 的執行快取與持久化模組權重。這讓第二次點擊 **執行** 時可以重用第一次 run 的權重與已快取的上游節點。若兩個 run *同時*共用這份狀態，會讀到彼此尚未建立完成的 tensor，因此伺服器會拒絕第二個 run。

這兩種拒絕都會立即明確回報（HTTP 503 或畫布上的錯誤），而不是無提示地等待。畫布前有使用者在等；點擊後若默默加入看不見的佇列，與當機沒有分別。實際使用時不會碰到每個 session 一個 run 的規則，因為 run 進行期間 **執行** 按鈕會停用。

## 觀察與取消

`GET /api/runs` 會為每個排隊中的 run 回報 `queue_position` — 從 1 開始，而且是**在它自己那條裝置佇列裡**的位置，所以排在兩個 CPU run 後面的 CPU run 就是第 3 位，不管它前面送出過幾個 CUDA run。`cdui run --wait` 等待時會印出這個位置，結果面板的 **執行任務** 分頁則會顯示成 `佇列第 N 位`。

取消一個還沒開始的 run，就只是把它從隊伍裡拿掉。什麼都沒執行、沒有碰到任何裝置，後面的 run 依序遞補。該筆紀錄會標成 `cancelled` 且沒有開始時間，正在追蹤它的客戶端也會收到一般的停止事件。

取消一個已經在跑的 run 則是協作式的 — 見[執行圖的「停止」一節](./running-graphs#停止)。

:::note 上限為 1 的佇列上，可能短暫出現兩個 running
run 的圖完成後會立即歸還裝置名額，再寫入最終狀態與結束事件。因此，下一個 run 可能在前一個 run 尚未完成狀態寫入時啟動，這段期間的查詢會在上限為 1 的佇列中看到兩筆 `running`。裝置上沒有重複執行任何內容；前一個 run 已經結束，裝置也已經可用。這項設計可避免 GPU 在資料庫寫入期間閒置，也可避免畫布收到 run 已結束的通知後，下一次執行仍因名額尚未歸還而被拒絕。
:::

### 執行任務面板 {/* #runs-panel */}

結果面板的 **執行任務** 分頁會列出伺服器持有的每個 run，不論它是從哪個分頁、`cdui run` 或 API 啟動，並依最新到最舊排列。篩選 chip 包含 **全部／執行中／排隊中／已成功／失敗／已取消／已中斷**，欄位是**名稱、狀態、裝置、開始、耗時、最終損失**。等待中的 run 會在裝置旁顯示 `佇列第 N 位`。每一列最多提供四個動作：

| 動作 | 用途 |
| --- | --- |
| **停止** | 要求排隊中或執行中的 run 停止 — 採協作式停止，見[停止](./running-graphs#停止)。 |
| **觀看** | 把該 run 串流到目前分頁的執行紀錄，並從頭重播。你可以用它查看從終端機或另一個分頁送出的 run；目前分頁會停止跟隨原本觀看的 run，但兩個 run 本身都不受影響。 |
| **CSV** | 下載該 run 的指標（`GET /api/runs/{id}/metrics?format=csv`）。 |
| **刪除** | 只適用於已結束的 run。移除該 run 的指標、事件紀錄、產出紀錄與任何錄製的輸出；磁碟上的 checkpoint 檔案會保留。 |

點擊一列可查看詳細資訊：有設定時會顯示亂數種子與**決定性**，失敗時會顯示錯誤；此外還有指標圖表及其**下載 CSV**、每筆已記錄產出檔案的**複製路徑**按鈕，以及事件紀錄的最後 200 筆。run 仍在進行時，紀錄會持續更新。若載入編輯器時仍有 run 在進行，畫面會顯示包含數量的通知，並引導你前往這個面板。

## `cdui run`

把存好的圖檔送到執行中的伺服器：

```bash
cdui run mygraph.json
```

這個指令會把進度串流到終端機，並在 run 結束時退出。它是**客戶端**：run 建立在伺服器上，因此即使關閉啟動它的終端機，run 仍會繼續執行。

```bash
# 命名、指定 GPU、指定種子
cdui run train.json --name "resnet epoch sweep" --device cuda:0 --seed 42

# 一次排五個工作然後就走
for i in 1 2 3 4 5; do cdui run "sweep-$i.json" --device cuda:0 --detach; done

# 跟著看，但十分鐘後就不等了（run 仍會繼續）
cdui run train.json --device cuda:0 --timeout 600

# 保留節點輸出供事後檢視
cdui run infer.json --record-outputs
```

```powershell
# 同一組批次送出的 PowerShell 版本
1..5 | ForEach-Object { cdui run "sweep-$_.json" --device cuda:0 --detach }
```

| 旗標 | 意義 |
| --- | --- |
| `--name <text>` | 存在 run 上的名稱，列出 run 的地方都會顯示 |
| `--device <dev>` | `cpu` \| `auto` \| `cuda` \| `cuda:N` \| `mps`（預設 `auto`，目前會解析成 `cpu`）。解析後的裝置就是它加入的佇列。 |
| `--seed <n>` | 用 `n` 為每個節點設定種子，讓執行可以重現。設了種子的執行會一次只跑一個節點 — 見 **[可重現的執行](./running-graphs#可重現的執行亂數種子)**。 |
| `--deterministic` | 同時要求 PyTorch 使用決定性運算核心（`warn_only`） |
| `--record-outputs` | 保留節點輸出供事後檢視 |
| `--wait` | 串流進度直到 run 結束（**預設**） |
| `--detach` | 印出 run id 後立刻以 0 離開 |
| `--timeout <s>` | 等待 N 秒後停止等待。run 仍在伺服器上繼續。 |
| `--host`、`--port` | 伺服器位址（預設沿用上次 `cdui start` 啟動的那台） |

run 在排隊時，CLI 會回報它排在第幾位，而不是沒有任何輸出：

```text
=== Run submitted ===
  Run ID          3f1c9ab2c04e4d5f8b1a7e6d2c930f45
  Graph           sweep-3.json
  Device          cuda:0
  Status          queued

  ○ queued  position 3 on cuda:0
  ▸ started
  ✓ dataset
    trainer  epoch 3/10  loss=0.1235
  ✓ trainer
  ✓ run complete

  Result          succeeded
```

### 離開碼 {/* #結束代碼 */}

| 代碼 | 意義 |
| --- | --- |
| 0 | run 成功（或 `--detach` 成功送出） |
| 1 | run 失敗、被取消或被中斷 — 或 CLI 無法送出它（沒有伺服器、找不到圖檔、envelope 被拒絕） |
| 2 | 命令列參數有問題（argument parsing） |
| 130 | Ctrl+C。只是停止**跟隨**，run 仍在伺服器上繼續執行，等同於當初就加了 `--detach`。 |

`--timeout` 到期同樣以 1 結束：指令不能替一個它已經不再觀察的 run 宣告成功。

:::tip 沒有伺服器？請改用離線執行器
`cdui run` 會連線到執行中的伺服器。若要在完全不啟動伺服器的情況下，直接於行程內執行圖形，請使用 **[CLI 圖形執行器](./cli-runner)**；這項工具沒有變動，適合沒有常駐服務的機器。
:::

## 參數掃描 {/* #sweeps */}

一次 **sweep** 會用不同的參數值重複執行同一張圖，再依一項指標為結果排名 — 也就是在佇列上做 grid 或 random search。`POST /api/sweeps` 會把搜尋空間編譯成每種組合各一份完整的圖，將每份圖當作一般 run 加入 queued 通道，然後以 `201` 回覆 `sweep_id`、`total_combinations`、展開後的 `params`，以及每個排隊中 variant 的一筆資料（`index`、`run_id`、`status`、`seed`、`params`）。三條 route 都列在 [API 參考](../advanced/api-reference)；兩個 `POST` 和其他會修改狀態的 route 一樣需要 session token，`GET` 則是開放的。

```json
{
  "base_graph": {"nodes": [ ... ], "edges": [ ... ]},
  "name": "lr x batch",
  "sweep_spec": {
    "method": "grid",
    "params": [
      {"node_id": "opt", "param": "lr",
       "range": {"min": 1e-4, "max": 1e-1, "count": 4, "scale": "log", "type": "float"}},
      {"node_id": "loader", "param": "batch_size", "values": [32, 64, 128]}
    ]
  },
  "objective": {"metric": "val_loss", "direction": "minimize"},
  "options": {"device": "cuda:0"}
}
```

`params` 裡的每一筆項目都以節點 id 指向該節點的一個參數，並帶有明確且不重複的 `values` 清單，或由系統展開的 `range`。`range` 會從 `min` 到 `max` 取 `count` 個點，並在 `linear` 或以 10 為底的 `log` 尺度上等距分布；`log` 尺度需要正的 `min`。若為 `type: int`，結果會四捨五入成整數；如果因此只剩較少的相異值，就只使用那些值。任何項目排入佇列前，每個值都會依節點定義檢查型別、允許的選項及 min/max。無法滿足的 spec 會收到指出該項目的 `400`，且不會留下只建立一部分的 sweep。

**Grid 或 random。** `method: grid` 會列舉每一種組合，最後列出的 param 變動最快，且不接受 `samples`。`method: random` 會抽取 `samples` 種不重複的組合，而且 `samples` 與 `seed`（0 到 4294967295）缺一不可；相同的 seed 永遠會抽到相同組合。要求的 sample 多於空間能提供的組合會被拒絕；編譯後 variant 數量超過上限的 sweep 也會被拒絕，絕不會默默截斷。

**`objective` 是必填的。** `metric` 是節點所記錄的一個 series 名稱（`train_loss`、`val_loss`、`eval_accuracy`，或外掛節點記錄的任何名稱），`direction` 則是 `minimize` 或 `maximize`。送出時不會檢查這個名稱，因為當時還沒有任何 variant 跑過；如果最後沒有任何 variant 記錄它，排名表會是空的，並附上一個 `objective_warning`，列出各 run 實際記錄的 series。

**`options`** 會原封不動交給每個 variant（device、`record_outputs` 等），但有三種情況會被拒絕：`options.seed`（seed 由 sweep 管理）、`lane: interactive`（sweep 一律排隊），以及 variant 數量超過輸出儲存上限（預設 20）時使用 `record_outputs`，因為最早幾個 variant 的捕獲資料會在 sweep 結束前被逐出。若要為訓練本身設定 seed，請設定 `sweep_spec.seed` 及 `"seed_variants": true`：第 *i* 個 variant 會使用 `seed + i`，超出時折回有效的 seed 範圍。若設定 `seed_variants: true` 卻沒有提供 `sweep_spec.seed`，請求會在建立任何資料列前以 `400` 拒絕。這會讓每個 variant 都成為 seeded run，因此一次只執行一個節點，也不會有其他 run 同時執行。seeded sweep 會嚴格依序執行，畫布 run 在整段期間都無法執行；詳見[可重現的執行](./running-graphs#可重現的執行亂數種子)。

**可以掃描什麼：** 已註冊節點上的 int、float、bool、string 與 select 參數。不支援 preset 實例的內部參數、subgraph 實例的參數、`SECRET` 參數（選中的值會以明文存入 sweep 資料列），也不支援在圖中出現兩次的 node id。

| 變數 | 預設 | 上限 |
| --- | --- | --- |
| `CODEFYUI_MAX_SWEEP_RUNS` | `32` | 單一 sweep 的 variant 數 — grid 大小或 `samples` |
| `CODEFYUI_MAX_SWEEP_PARAMS` | `4` | `params` 的項目數 |
| `CODEFYUI_MAX_SWEEP_DOMAIN` | `32` | 單一 param 在 range 展開後的相異值數量 |

### 讀取結果 {/* #reading-the-results */}

`GET /api/sweeps/{id}` 會回傳該 sweep：它的 `state`（`running`、`cancelling`、`finished`；若送出迴圈中途失敗則為 `failed`，但已排入佇列的 child 仍會繼續）、objective、依 status 分組的 `counts`、包含每個展開後 domain 的 `params`，以及按**排名順序**、最佳者優先的 `variants`。每個 variant 會帶著自己的 `index`（送出順序）、`run_id`、即時 `status`、收到的 `params`、`seed`、達到的 `objective` 值、`rank`、`run_exists`；child run 仍存在時另有其 `final_metrics`。`best` 指向排名第 1 的 variant。run 結束且記錄過 objective 後，variant 會依該 series 的最終值排名；run 即使在記錄後失敗仍會列入排名，沒有排名的 variant 則保留自己的資料列，以 index 順序排列並顯示 `rank: null`。`?format=csv` 可將同一份表格下載成試算表，每個掃描參數各佔一欄。

即使 child run 被移除，結果仍會保留：每個已結束 variant 的 objective 都會複製到 sweep 資料列上，而 retention 會在刪減 run 前收回任何尚未讀取的結果。被刪減或手動刪除的 child 會顯示 `status: "missing"` 與 `run_exists: false`，但該資料列仍會保留。

### 取消 {/* #cancelling */}

`POST /api/sweeps/{id}/cancel` 會要求每個排隊中或執行中的 child 停止 — 每個各送一次協作式取消 — 並回報 `cancelled` 與 `already_finished` 的數量，以及按 index 排列的逐 variant 清單。只有至少一個 child 當時仍在活動時，sweep 的 state 才會變成 `cancelling`；它永遠不會變成 `cancelled`：前 30 個 variant 完成、最後 2 個停止的 sweep，是一個帶有 2 列 cancelled 的 finished sweep。所有 child 都進入終止狀態後，下一次讀取（如果它們原本都已終止，也就是 cancel 回覆本身）會把 state 確定為 `finished`。重複要求不會有副作用（`cancelled: 0`）。

### variants 會出現在哪裡 {/* #where-the-variants-show-up */}

編輯器目前還沒有 sweep 專屬畫面。child 都是一般 run：它們會以 sweep 的 `name` 出現在[執行任務面板](#runs-panel)，而 `GET /api/runs` 的每一列都帶有 `sweep_id` 與 `sweep_variant`。要跟隨某一個 variant，可像其他 run 一樣使用 `GET /api/runs/{id}/events`。

## 伺服器停止時

佇列不會在伺服器重啟後繼續執行。排程只存在伺服器記憶體中；如果保留等待中的資料列，它會持續等待已不存在的排程器。

正常停止（`cdui stop`）會立即將每個等待中的 run 標記為 `interrupted`、寫入一般停止事件，並要求執行中的 run 以協作方式停止。強制終止行程，或執行中的工作超過正常停止的寬限時間，都可能留下 `queued` 或 `running` 資料列。下次啟動時，復原程序會把這兩種狀態都改成 `interrupted`；兩者都不會繼續執行。

若仍要執行這些工作，請重新送出。

## 設定

以下都是啟動時讀取的環境變數（跟其他設定一樣使用 `CODEFYUI_` 前綴）。

| 變數 | 預設 | 意義 |
| --- | --- | --- |
| `CODEFYUI_RUN_QUEUE_MAX_CONCURRENT_GPU` | `1` | 每條加速器佇列的並行 run 數 |
| `CODEFYUI_RUN_QUEUE_MAX_CONCURRENT_CPU` | `2` | `cpu` 佇列的並行 run 數 |
| `CODEFYUI_RUN_QUEUE_MAX_CONCURRENT` | *(空)* | 個別 key 的覆寫，例如 `cuda:0=2,cpu=8`。優先於上面兩個預設值。 |
| `CODEFYUI_RUN_INTERACTIVE_MAX_CONCURRENT` | `2` | 畫布 run 的並行上限 |
| `CODEFYUI_MAX_PARALLEL_NODES` | `4` | 單一 run 內同時執行的節點數 |
| `CODEFYUI_RUN_RETENTION_KEEP_LAST` | `200` | 保留幾筆已結束的 run。`0` 表示完全不保留；負值會停用 retention。 |
| `CODEFYUI_RUN_EVENT_PAYLOAD_CAP_BYTES` | `131072` (128 KB) | 儲存及分送的單一事件 payload 上限；超過上限的輸出項目會換成省略標記，但仍會指出它的 port。`0` 或更小的值會停用上限。 |
| `CODEFYUI_RUN_EVENTS_RESPONSE_CAP_BYTES` | `4194304` (4 MB) | 單一 `GET /api/runs/{id}/events` page 的 byte 預算；提早停止的 page 會回傳 cursor，供下一次接續。 |

格式錯誤的覆寫項目會被忽略並記錄警告，不會讓伺服器啟動失敗。小於 1 的上限一律視為 1，避免佇列因無法排空而停滯。三項 sweep 上限列在[參數掃描](#sweeps)一節。

Retention 會在啟動時及每個 run 結束後執行，保留最新的 `KEEP_LAST` 筆已結束 run。活動中的 run 不會被刪減，但仍會計入保留窗口。和執行任務面板的**刪除**不同，刪減也會移除該 run 自動寫出的 checkpoint 與 TensorBoard 目錄。結果為 `interrupted` 的 run 所留下的 checkpoint 會保留，以避免 crash 造成 resume point 遺失。

## 延伸閱讀

- [執行圖](./running-graphs) — 單一 run 內部發生了什麼
- [CLI 圖形執行器](./cli-runner) — 不開伺服器執行一張圖
- [API 參考](../advanced/api-reference) — `/api/runs` 端點
