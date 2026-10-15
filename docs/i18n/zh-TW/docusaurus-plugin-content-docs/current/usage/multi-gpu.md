---
sidebar_position: 3.6
title: 多 GPU 執行
description: 在多 GPU 的機器上使用 CodefyUI——何時排入佇列而不是按 Run、在同一台機器上啟動 worker agent、選擇裝置或裝置集合、slot 與上限、Workers 面板、畫布的行為、峰值記憶體，以及目前的限制。
---

# 多 GPU 執行

CodefyUI **伺服器**是一個程序。**run** 預設在這個程序內執行；指定 worker agent 為目標時，則在該 worker 的子程序中執行，並使用目標的裝置。在有多張 GPU 的機器上，這帶來兩件事：多個 run 同時執行，每張卡一個（若你允許，也可以更多）；以及單一 run 看得到多張卡。本頁是操作說明；設計見[執行架構](/advanced/execution-architecture)，佇列順序與選項的參考見[執行佇列](./run-queue)。

## 直接執行或排入佇列 {/* #run-or-queue */}

| | **Run**（畫布） | **Queued run** |
| --- | --- | --- |
| 從哪裡啟動 | 工具列的 **Run** | Run 旁的**排入佇列...**、`cdui run`、`POST /api/runs`、sweep 或 study |
| 在哪裡執行 | 一律在伺服器程序內 | 伺服器程序內，或 `target` 指定的 worker |
| 是否等待空出的卡 | 否：立即開始（最多兩個，每個分頁一個） | 是：加入其裝置的佇列 |
| 裝置 | 一個：圖的裝置，否則為 Settings 的裝置 | 一個裝置，或 worker 上的一組裝置集合 |
| 在多次點擊間保留權重與快取、為 Inspector 擷取輸出 | 是 | 否（只有 `record_outputs` 才擷取輸出，且只限伺服器本身） |

建構與除錯圖時用 **Run**。耗時長的工作、想放在特定卡上的工作，以及需要不只一張卡的工作，請排入佇列。

### 排入佇列... {/* #queue-on */}

工具列 **Run** 旁的下拉鍵會開啟選單，其中的**排入佇列...**會把目前分頁的圖送出為 queued run（與 `cdui run` 相同的 `POST /api/runs`）：

- **目標**：**本伺服器**、每個已連線的 worker 及其狀態與裝置（排空中的 worker 會列出但不能選），以及有 worker 連線後出現的**任一可用**。
- **裝置**：本伺服器只能選一個裝置；worker 可選同一種類的一組裝置，依 worker 列出裝置的順序排列，勾選 CUDA 卡時會顯示它在 run 內的名稱（`cuda:2 -> cuda:0 in this run`）；任一可用則選裝置種類（CPU、CUDA GPU、Apple MPS）。
- **名稱**、**亂數種子**、**確定性演算法**與**錄製節點輸出**的初始值取自分頁設定。目標為 worker 或任一可用時，錄製節點輸出會停用。

**排入佇列**會送出 run、關閉對話框，並在 Runs 分頁開啟它。伺服器拒絕時，訊息會顯示在對話框內。

## 在這台機器上啟動 worker {/* #start-a-same-host-worker */}

worker agent 會把它接到的每個 run 放在自己的子程序中執行，所以崩潰或記憶體不足被終止只會結束那一個 run，伺服器與其他 run 不受影響。這個版本的 agent 與伺服器在同一台機器上執行。

```bash
cdui start                 # the server, if it is not running yet
cdui agent start           # a worker named after this machine's hostname
cdui agent stop            # when you are done
```

在 `cdui start` 旁執行 `cdui agent start` 不需要任何參數：它會自己找到伺服器的位址與 session token，伺服器重啟後也會重新連線。`--name gpu-box` 指定 worker 名稱，`--devices cuda:0,cuda:1` 限制它提供的卡（CPU 一律提供），`--foreground` 讓它留在前景供 systemd 使用。所有參數見 [CLI 指令：Agent 指令](/getting-started/cli-commands#agent-commands)。

幾秒內 worker 就會出現在 [Workers 面板](#the-workers-panel)與 `GET /api/workers`。

## 選擇裝置 {/* #choose-devices */}

queued run 可以在[排入佇列...](#queue-on)對話框、`cdui run` 的 `--target` 與 `--devices`，或 `POST /api/runs` 的 `options.target` 指定執行位置：

```bash
# One card on this server, as before
cdui run train.json --device cuda:1

# One card on the worker gpu-box
cdui run train.json --target gpu-box --device cuda:1

# Two cards on the worker: the run sees them as cuda:0 and cuda:1
cdui run train.json --target gpu-box --devices cuda:2,cuda:3

# Whichever has a free CUDA card first: a connected worker, else this server
cdui run train.json --target any --device cuda
```

```json
{"graph": {...}, "options": {"target": {"worker": "gpu-box", "devices": ["cuda:2", "cuda:3"]}}}
```

- **`local`**（預設）是這台伺服器。在這裡 run 只在一個裝置上執行：帶裝置集合時，它占用集合中每個裝置的 slot，因此不會有其他 run 被放到這些裝置上，並在第一個裝置上執行。
- **worker 名稱**把 run 送到該 worker。該 worker 必須在伺服器啟動後連線過並提供這些裝置，否則送出會以 `400` 失敗。目前離線的 worker 不會造成拒絕：run 會等它重新連線。
- **`any`** 依名稱順序選第一個已連線、未排空且有空 slot 的 worker，這台伺服器排最後。帶裝置集合的 `any` run 只會在 worker 上執行，因為只有 worker 會為該 run 把集合從 `cuda:0` 重新編號（見下文）；伺服器見過的 worker 都不提供整個集合時，送出會以 `400` 被拒絕。

**run 內的裝置名稱。** worker 上的 run 只看得到自己的卡，並依你列出的順序從 0 重新編號。以 `--devices cuda:2,cuda:0` 為例，run 的 `cuda:0` 是機器的 `cuda:2`，run 的 `cuda:1` 是機器的 `cuda:0`。**device** 參數為 `auto` 的節點跟隨 run 的裝置（第一張卡）；要把節點放到集合中的另一張卡，把它的 **device** 設為那張卡在 run 內的名稱，此例為 `cuda:1`。run 會把對應關係記在 `execution_facts` 的 `visible_devices` 與 `device_aliases`。請參閱[裝置後端](/advanced/device-backends#addressing-one-card-out-of-several)。

**Sweep 與 study。** `POST /api/sweeps` 與 `POST /api/studies` 會把 `options.target` 傳給每個子 run。使用 `"target": {"worker": "any"}` 且不帶 `devices` 時，每個子 run 取下一個符合 sweep 裝置種類的空 slot，所以在兩個各有一張卡的 agent 上，四個 variant 的 sweep 會同時跑兩個；worker 上帶種子的子 run 各在自己的程序中，可以並行。單一 run 會被拒絕的 target，會讓整個 sweep 或 study 在排入任何東西之前以 `400` 被拒絕。每個 variant 與 attempt 都會回報 `worker_id`、`devices` 與 `queue_key`。**新增掃描**對話框有與 **Queue on...** 相同的目標與裝置選擇器：本伺服器、已連線的 worker 加上裝置集合，或任一可用加上裝置種類。維持本伺服器與分頁的裝置時，sweep 會像以往一樣送出，不帶 target。sweep 檢視會顯示每個 variant 的 Worker 與裝置。study 請透過 API 設定 `target`。請參閱[執行佇列：Sweep](./run-queue#sweeps)。

## Slot 與上限 {/* #slots-and-limits */}

**slot** 是一台機器上的一個裝置。加速器的每個 slot 一次容納一個 run，CPU 容納兩個，其餘依送出順序等待。裝置集合 run 需要集合中每個裝置都有空 slot，並一次全部占用；它在佇列最前面等待時會保留這些 slot，所以一連串單卡 run 無法讓它餓死。

`CODEFYUI_RUN_QUEUE_MAX_CONCURRENT` 修改單一 slot 的上限。只寫裝置指的是這台伺服器；`worker/device=N` 指的是某個 worker：

```bash
CODEFYUI_RUN_QUEUE_MAX_CONCURRENT="cuda:0=2,gpu-box/cuda:1=3" cdui start
```

同一台機器上的 worker，對實體上相同的每張卡（以卡的 UUID 比對）共用這台伺服器的 slot，因此在上限為 1 時，伺服器上的 run 與該 worker 上的 run 不會同時拿到同一張卡。這種卡套用伺服器的上限，針對它的 `worker/device` 覆寫會被忽略並發出警告。提高上限會讓多個 run 共用一張卡的記憶體，但沒有任何機制檢查它們是否放得下。細節見[執行佇列：設定](./run-queue#configuration)。

## Workers 面板 {/* #the-workers-panel */}

從 Runs 面板標題列的 **Workers**，或 **Settings → 執行 → Workers → 管理**開啟。開啟期間每 5 秒更新一次，每個 worker 會顯示：

- 狀態（**Online**、心跳延遲時為 **Degraded**、**Draining**、**Disconnected**）、版本以及是否與伺服器相同、環境摘要、mount 標籤，以及上次心跳距今多久；
- 每個裝置在最近一次心跳時的可用與總記憶體，以及 slot 的已占用 / 上限；
- 它正在執行的 run，每個都連到 Runs 面板中的該 run。

**排空**讓新的 run 不再被放到該 worker，它正在執行的 run 則會跑完；**取消排空**恢復原狀。worker 正在排空或離線時，queued run 會等待，Runs 面板會在該 run 的 worker 下方說明原因。**新增 worker** 顯示上面那段同機指令與其參數。

## 畫布的行為 {/* #the-canvas */}

畫布上的 **Run** 一律在伺服器程序內、以圖的裝置（否則為 Settings 中的裝置）執行，從不送到 worker。它跳過佇列，所以即使某張卡的上限是 1，它仍可能與 queued run 共用那張卡。它會在多次點擊之間保留分頁的快取輸出與模型權重，並為 Inspector 記錄輸出，這些 worker run 都做不到。要把分頁中的圖交給 worker，請使用[排入佇列...](#queue-on)。

## Runs 面板中的峰值記憶體 {/* #peak-memory */}

run 結束後，詳細資訊標題會顯示 CUDA 的**峰值記憶體**（配置器的最高水位，滑鼠停留可看到每張卡），或 MPS 的**結束時佔用記憶體**（驅動程式仍持有的量，含快取；MPS 沒有峰值計數器）。worker 上的 run 在自己的子程序中單獨執行，因此數字是精確的。在伺服器程序內執行的 run 共用一個 CUDA context，若同時有另一個 run 使用同一張卡，數字只是近似值。提高某張卡的上限之前，可以用已結束 run 的峰值判斷兩個這樣的 run 是否放得進一張卡。

## 限制 {/* #limits */}

- **單一 run 不會跨卡訓練。** 裝置集合讓一個 run 看得到多張卡，節點也可以用 run 內名稱固定到這些卡上，但目前還沒有 `DistributedDataParallel` 策略（[#792](https://github.com/CodefyUI/CodefyUI/issues/792)）。
- **沒有依記憶體的准入。** run 以每個 slot 的數量准入；不會因為卡上記憶體不足而拒絕 run，塞太滿的卡會以記憶體不足錯誤失敗（[#788](https://github.com/CodefyUI/CodefyUI/issues/788)）。請參閱[訓練記憶體](/advanced/training-memory)。
- **只限同一台機器。** 遠端 GPU 伺服器需要 enrollment（[#780](https://github.com/CodefyUI/CodefyUI/issues/780)）。
- **worker 上不擷取輸出**（`record_outputs`，[#789](https://github.com/CodefyUI/CodefyUI/issues/789)），也不保留權重：每個 worker run 都從頭開始。
- **斷線會結束 run。** agent 停止或失去連線時，它的 run 以 `interrupted` 結束（[#783](https://github.com/CodefyUI/CodefyUI/issues/783) 會加入續跑）。

## 另請參閱 {/* #see-also */}

- [執行佇列](./run-queue)——佇列順序、`cdui run`、`target` 選項與 worker 的完整說明
- [執行架構](/advanced/execution-architecture)——控制面、executor、worker 協定與不變量
- [裝置後端](/advanced/device-backends)與[訓練記憶體](/advanced/training-memory)——裝置、指定單張卡與記憶體不足錯誤
