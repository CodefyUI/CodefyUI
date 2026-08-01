---
sidebar_position: 3.5
title: 執行佇列
description: 每個裝置一條佇列，丟著就走 — 通道、FIFO 順序、並行上限、取消排隊中的 run，以及 cdui run 指令。
---

# 執行佇列

一個 run 屬於**伺服器**，不屬於送出它的人。關掉分頁、關掉 terminal、登出都沒關係 — run 會繼續跑，之後可以從 Runs 面板或 CLI 回來看它。

這正是佇列有用的原因。一次送出五個訓練工作、闔上筆電，伺服器會在 GPU 上一個一個做完，而不是五個同時開跑、四十分鐘後才因為 VRAM 不足而全部倒掉。

## 每個裝置一條佇列

每個 run 都依它**解析後的裝置**排程 — `cpu`、`cuda:0`、`mps`。這個字串就是佇列的 key，所以每個裝置有自己獨立的隊伍與自己的並行上限。

| 佇列 key | 同時執行數（預設） | 為什麼 |
| --- | --- | --- |
| `cuda:0`、`cuda:1`、`mps`，任何加速器 | 1 | 同一張卡上的第二個 run 會跟第一個搶同一份 VRAM。猜錯的典型下場，是長時間工作跑到一半才出現 CUDA out-of-memory。 |
| `cpu` | 2 | CPU 的失敗模式是變慢，不是直接死掉 — 核心夠多的機器就該用起來。 |

塞滿的 GPU 不會拖到 CPU 的 run，兩張不同的卡也不會互相等待。同一條佇列內的 run 依**送出順序**啟動，先到先跑。

:::note `--device auto` 目前會解析成 CPU
裝置解析會把未知或 `auto` 的請求對應到 `cpu`，所以 `auto` 現在是排在 CPU 那條線上。要用顯示卡請明確指定 `cuda`（或 `cuda:0`）。
:::

這是**每個裝置能跑幾個 run** 的上限，和 `CODEFYUI_MAX_PARALLEL_NODES` 不是同一個旋鈕 — 後者限制的是*單一 run 內部*同時執行幾個節點。兩者是相乘的關係。

## 通道（lane）

一個 run 的**通道**記錄它從哪裡來，並決定它怎麼被排程。

| 通道 | 由誰送出 | 排程方式 | 上限 |
| --- | --- | --- | --- |
| `queued`（預設） | `POST /api/runs`、`cdui run` | 加入該裝置的 FIFO | 依裝置，見上表 |
| `interactive` | 畫布（在圖上按 **Run**） | 跳過 FIFO，立即開始 | 同時 2 個，且每個編輯連線只能有 1 個 |

畫布跳過佇列是刻意的：課堂上的示範不該排在六小時的訓練工作後面。代價是 interactive 的 run 可能讓一張卡超出該裝置的 queued 上限 — FIFO 的用意是讓無人看顧的工作有秩序，而不是讓某個裝置被獨佔。

### interactive 通道的兩道上限

因為它跳過佇列，interactive 通道改用另外兩種方式設限：

- **最多兩個 interactive run 同時執行**（`CODEFYUI_RUN_INTERACTIVE_MAX_CONCURRENT`），這樣一整排開著的分頁不會把 GPU 吃光。
- **每個編輯連線只能有一個 run。** 每個開著的畫布連線會把自己的執行快取與持久化的模型權重借給它啟動的 run — 這正是連按兩次 **Run** 能沿用前一次權重與上游快取節點的原因。兩個 run *同時*共用這份狀態會讀到彼此做到一半的 tensor，所以伺服器會拒絕第二個。

兩種拒絕都是立即而明確的（HTTP 503，或畫布上的錯誤訊息），而不是默默等待。畫布前面坐著一個活生生的使用者，而一次「悄悄加入看不見的佇列」的點擊，跟當掉是分不出來的。實務上你不會碰到每連線一個的規則：run 進行中時 **Run** 按鈕本來就是停用的。

## 觀察與取消

`GET /api/runs` 會為每個排隊中的 run 回報 `queue_position` — 從 1 開始，而且是**在它自己那條裝置佇列裡**的位置，所以排在兩個 CPU run 後面的 CPU run 就是第 3 位，不管它前面送出過幾個 CUDA run。Runs 面板直接顯示這個數字。

取消一個還沒開始的 run，就只是把它從隊伍裡拿掉。什麼都沒執行、沒有碰到任何裝置，後面的 run 依序遞補。該筆紀錄會標成 `cancelled` 且沒有開始時間，正在追蹤它的客戶端也會收到一般的停止事件。

取消一個已經在跑的 run 則是協作式的 — 見 [執行圖](./running-graphs) 的「停止」一節。

## `cdui run`

把存好的圖檔送到執行中的伺服器：

```bash
cdui run mygraph.json
```

這個指令會把進度串流到終端機，並隨著 run 結束而結束。它是個**客戶端**：run 建立在伺服器上，所以會比啟動它的 terminal 活得久。

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
| `--name <文字>` | Runs 面板顯示的名稱 |
| `--device <裝置>` | `cpu` \| `auto` \| `cuda` \| `cuda:N` \| `mps`（預設 `cpu`）。解析後的裝置就是它加入的佇列。 |
| `--seed <n>` | `random`、NumPy 與 torch 的隨機種子 |
| `--record-outputs` | 保留節點輸出供事後檢視 |
| `--wait` | 串流進度直到 run 結束（**預設**） |
| `--detach` | 印出 run id 後立刻以 0 離開 |
| `--timeout <秒>` | 等待上限。逾時後 run 仍在伺服器上繼續。 |
| `--host`、`--port` | 伺服器位址（預設沿用上次 `cdui start` 啟動的那台） |

run 在排隊時，CLI 會告訴你排到第幾位，而不是靜靜不動：

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

### 結束代碼

| 代碼 | 意義 |
| --- | --- |
| 0 | run 成功（或 `--detach` 成功送出） |
| 1 | run 失敗、被取消或被中斷 — 或 CLI 根本沒送出去（伺服器沒開、找不到圖檔、被伺服器拒絕） |
| 2 | 命令列參數有問題（argparse） |

`--timeout` 到期同樣以 1 結束：指令不能替一個它已經不再觀察的 run 宣告成功。

:::tip 沒有伺服器？請改用離線執行器
`cdui run` 需要一台執行中的伺服器。若要完全不開伺服器、直接在行程內執行一張圖，請用 **[CLI 圖形執行器](./cli-runner)** — 它沒有變動，仍然是沒有常駐服務的機器上的正確工具。
:::

## 伺服器停止時

佇列不會跨重啟續跑：排程只存在伺服器的記憶體裡，留著一筆排隊中的紀錄只會讓它永遠等一個已經不存在的排程器。

所以正常停止（`cdui stop`）會在離開前把每個排隊中的 run 標成 `interrupted`，並寫入一般的停止事件。強制砍掉行程則會讓那些紀錄停在 `queued`，而下次啟動時同樣會把它們標成 `interrupted`。兩條路徑產生的結果一樣，正常停止只是立刻做完。至於*已經在執行*的 run，會先協作式地停下來，並記錄自己誠實的狀態。

還想跑的東西，重新送出一次即可。

## 設定

以下都是啟動時讀取的環境變數（跟其他設定一樣使用 `CODEFYUI_` 前綴）。

| 變數 | 預設 | 意義 |
| --- | --- | --- |
| `CODEFYUI_RUN_QUEUE_MAX_CONCURRENT_GPU` | `1` | 每條加速器佇列的並行 run 數 |
| `CODEFYUI_RUN_QUEUE_MAX_CONCURRENT_CPU` | `2` | `cpu` 佇列的並行 run 數 |
| `CODEFYUI_RUN_QUEUE_MAX_CONCURRENT` | *(空)* | 個別 key 的覆寫，例如 `cuda:0=2,cpu=8`。優先於上面兩個預設值。 |
| `CODEFYUI_RUN_INTERACTIVE_MAX_CONCURRENT` | `2` | 畫布 run 的並行上限 |
| `CODEFYUI_RUN_RETENTION_KEEP_LAST` | `200` | 保留幾筆已結束的 run |

格式錯誤的覆寫項目只會被忽略並留下警告，不會讓伺服器起不來；小於 1 的上限一律當成 1 — 一條永遠排不完的佇列是當掉，不是設定。

## 延伸閱讀

- [執行圖](./running-graphs) — 單一 run 內部發生了什麼
- [CLI 圖形執行器](./cli-runner) — 不開伺服器執行一張圖
- [API 參考](../advanced/api-reference) — `/api/runs` 端點
