---
sidebar_position: 6.5
title: 執行架構
description: run 可以在伺服器程序內或 worker agent 上執行時，從送出到結果的路徑——控制面、executor、worker agent 與 run 子程序、slot 與 placement、事件路徑、隔離、儲存、安全，以及每項變更都要遵守的不變量。
---

# 執行架構

[架構](./architecture)說明整個伺服器。本頁深入其中一部分：run 的圖在哪裡執行。預設情況下，每個 run 都在伺服器程序內執行，與 3.0 完全相同。queued run 也可以改送到 **worker agent**（`cdui agent start`），由它在自己的子程序中執行。設計追蹤於 [#764](https://github.com/CodefyUI/CodefyUI/issues/764)；使用方式見[執行佇列：Workers](/usage/run-queue#workers) 與 [CLI 指令：Agent 指令](/getting-started/cli-commands#agent-commands)。

:::note 現況
程式碼已有：executor 接縫、依 `(worker, device)` slot 放置、`target` 選項（單一 run、sweep 與 study 皆可用）、與伺服器在同一台機器上、或以一次性 token 加入的另一台機器上的 worker agent（[#780](https://github.com/CodefyUI/CodefyUI/issues/780)）、握手時檢查版本、環境指紋與節點目錄（[#781](https://github.com/CodefyUI/CodefyUI/issues/781)）、儲存 mount（[#782](https://github.com/CodefyUI/CodefyUI/issues/782)）、斷線與續接（[#783](https://github.com/CodefyUI/CodefyUI/issues/783)）、SSH 反向通道（[#784](https://github.com/CodefyUI/CodefyUI/issues/784)）、transfer 後端（[#785](https://github.com/CodefyUI/CodefyUI/issues/785)）、agent 對控制面 TLS 憑證的驗證（[#786](https://github.com/CodefyUI/CodefyUI/issues/786)）、裝置集合、每個 run 的峰值記憶體、Workers 面板，以及工具列的 **Queue on...** 對話框。遠端 worker 的設定方式見[遠端 Worker](/usage/remote-workers)。
:::

## 名詞 {/* #vocabulary */}

| 名詞 | 意義 |
|------|---------|
| **控制面**（control plane） | `cdui start` 的伺服器程序：UI、REST、`/ws/execution`、SQLite、run service 與 worker registry。一個程序。 |
| **Executor** | 執行一個 run 的圖並回報結束方式的物件（`core/executors.py`）。`InProcessExecutor` 在控制面內執行；`WorkerExecutor` 交給 worker 執行。 |
| **Worker**（worker agent） | 一個 `cdui agent start` 程序。它主動連線到控制面、接收指派，並為每個 run 啟動一個 run 子程序。它本身從不 import PyTorch。 |
| **Run 子程序** | worker 為單一 run 以 `spawn` 啟動的全新直譯器。在 worker 上，只有這裡會 import PyTorch 並初始化 CUDA。 |
| **Slot** | 一台機器上的一個裝置，鍵為 `(worker_id, device)`：`("local", "cuda:0")` 是伺服器的第一張卡，`("gpu-box-1", "cuda:0")` 是 worker `gpu-box-1` 的第一張卡。一個 slot 同時容納固定數量的 run。 |
| **Placement** | 排程器對等待中 run 的回答：哪個 worker、在哪些裝置上執行、占用哪些 slot。 |
| **Target** | 送出時帶的 `options.target` 物件：`{"worker", "devices"}`。 |
| **裝置集合** | 列出同一種類、多於一個裝置的 `target.devices`。 |
| **Mount** | 兩台機器共用、各自位於不同路徑之資料夾的標籤（`data`）（[儲存](#storage)）。 |
| **Transfer 後端** | 由控制面擔任檔案庫，供與它沒有共用資料夾的 worker 使用（[儲存](#storage)）。 |

## 角色 {/* #roles */}

**控制面。** `cdui start` 的伺服器程序擁有所有需要持久保存的東西，以及畫布需要的一切。`RunService`（`core/run_service.py`）接收送出、為每個 queue key 維護一條 FIFO、詢問排程器哪個 run 可以開始、寫入 run 列與事件紀錄，並把事件分送給 `/ws/execution` 與 REST 讀取端。`WorkerRegistry`（`core/workers.py`）在記憶體中保存已連線的 worker：裝置、心跳、狀態、各自正在執行的 run，以及聯絡它的 socket。控制面重啟後沒有任何 worker，每個 agent 會在下次連線時重新註冊。

**Executor。** `RunService._execute` 建立一個 `ExecutionRequest`（圖、選項、`ExecutionContext`、placement、sink），並 await `Executor.run`，後者回傳 `ExecutionOutcome`（狀態、錯誤、峰值記憶體）。由 placement 決定使用哪個 executor：

- `InProcessExecutor` 在控制面程序內呼叫 `execute_graph`。它就是 #766 之前 `_execute` 的本體，未經修改；沒有 target 或 target 為 `local` 的 run 都由它執行。
- `WorkerExecutor` 送一個 `assign` frame 給被放置的 worker，並把回傳的 frame 轉成與 `InProcessExecutor` 相同的 sink 呼叫。對 run 的停止會轉送為 `cancel`。

**Worker agent。** `cdui agent start`（`backend/app/agent/`）是一個小型 asyncio 程序。`connection.py` 連線到 `/ws/worker`、送出 `hello`、每 `welcome.heartbeat_s` 秒（預設 10）送出心跳，並分派 `assign`、`cancel` 與 `drain`。`runner.py` 啟動並監看 run 子程序。agent 透過 NVML 讀取裝置資訊而不 import torch（`core/resources.py` 的無 torch 模式），因此它不可能持有自己的 CUDA context。

**Run 子程序。** `child.py` 是子程序的進入點。它在任何東西 import torch 之前套用 agent 選定的環境（`CUDA_VISIBLE_DEVICES` 等，見[隔離](#isolation)），以 `RunService._start` 的方式建立 `ExecutionContext`，再用寫入 pipe 的 sink 執行 `InProcessExecutor`。agent 在一條執行緒上讀這條 pipe，把每則訊息轉成協定 frame 送出。

```
Browser / cdui run ── REST, /ws/execution ──► control plane (cdui start)
                                               RunService · scheduler · WorkerRegistry · SQLite
                                               InProcessExecutor (default) · WorkerExecutor
                                                          ▲
                                                          │ WS /ws/worker  (the agent dials out)
                                                          │
                                               cdui agent (same machine, this release)
                                               one spawn subprocess per run
                                               CUDA_VISIBLE_DEVICES = the run's cards
```

## 通道與執行位置 {/* #lanes-and-where-they-execute */}

每個 run 屬於兩條通道之一（[執行佇列：通道](/usage/run-queue#lanes)）：

- **Interactive**（畫布）。它一律在控制面程序內執行，因為它使用存在那裡的狀態：socket 的執行快取、持久化的模組權重，以及 Inspector 的擷取輸出。把 interactive 通道與 worker target 一起送出，會在送出時被拒絕。
- **Queued**（`POST /api/runs`、`cdui run`、工具列的 **Queue on...** 對話框、sweep、study）。預設在控制面執行。若 `options.target.worker` 設為某個 worker 名稱，或設為 `any` 並被放到某個 worker 上，就在該 worker 的 run 子程序中執行。sweep 或 study 會把它的 `options.target` 傳給每個子 run，因此 `any` 會把子 run 分散到空出的 slot。worker run 在 [#789](https://github.com/CodefyUI/CodefyUI/issues/789) 之前不接受 `record_outputs`；要擷取輸出請以 `local` 為 target。

沒有 `target`、或 `"worker": "local"` 的 run 完全不會經過 worker 程式路徑：queue key、`InProcessExecutor` 與事件序列都與 3.0.2 相同，`single-node-parity` CI 工作會拿它與儲存的快照比對。

## 事件路徑 {/* #event-path */}

不論圖在哪裡執行，事件都只建立一次、儲存一次。

1. **引擎 callback。** `execute_graph` 透過三個 callback 回報進度、方塊內部狀態與節點訊號（指標、產物、警告）。
2. **純函式建構器。** `core/run_events.py` 把 callback 參數轉成事件 payload（`node_status`、`inner_node_status`、指標點、訊號 frame）。這些函式沒有 I/O，也不引用 `RunService`，所以 run 子程序可以 import 它們並產生相同的位元組。
3. **Sink。** `RunSinks` 帶有四個 callable：`emit`（進入 `RunService._emit`，以 `CODEFYUI_RUN_EVENT_PAYLOAD_CAP_BYTES` 限制 payload 大小後寫入 `exec_run_events`）、`record_metrics`（`exec_run_metrics` 背後的批次器）、`record_artifact`（`exec_run_artifacts`）與 `warn`（一個 `run_warning` 事件）。
4. **持久紀錄。** 事件紀錄就是紀錄本身。`/ws/execution` 與 `GET /api/runs/{id}/events` 都是它的視圖；重新連線的客戶端從自己的 cursor 重播。

worker 上的 run，第 1、2 步發生在 run 子程序。它的 sink 把 `("event", ...)`、`("metrics", ...)`、`("artifact", ...)`、`("warning", ...)`，最後是 `("finished", ...)` 寫進 pipe；agent 把每一則轉成 frame，以每個 run 各自的 `seq` 編號，經 `/ws/worker` 送出；`WorkerExecutor` 再把每個 frame 交給控制面真正的 sink。只有節點層級的詞彙會傳送：`execution_start` 與終止事件（`execution_complete`、`execution_error`、`execution_stopped`）仍由控制面產生，與 in-process run 相同。run 子程序以控制面自己的規則限制 payload 大小，控制面再套用一次，結果不變。

frame 只在 `core/worker_protocol.py` 定義一次，兩端共用：

| 方向 | Frame |
|-----------|--------|
| agent → 控制面 | `hello`（名稱、版本、協定版本、環境指紋、節點目錄雜湊與節點型別、capabilities、machine id、裝置、mount、平台）、`heartbeat`（含可用記憶體的裝置、執行中的 run id）、`run_alive`（重新連線後），以及每個 run 的 `run_started`、`event`、`metrics`、`artifact`、`warning`、`run_finished` |
| 控制面 → agent | `welcome`（worker id、`heartbeat_s`、`grace_s`、capabilities、選用的 `resume_grace_s`：控制面保留已斷線 worker 之 run 的時間，agent 的上傳重試依此計算）、`reject`（原因，然後關閉）、`assign`（run id、圖、選項、device、devices、選用的 `grace_s`）、`cancel`（run id、選用的 `grace_s`）、`drain`、`ack`（run id、seq：該 run 到這個 seq 為止的 frame 都已記錄） |

## 排程 {/* #scheduling */}

**Queue key 與 slot。** 每個等待中的 run 都在一條以字串為鍵的 FIFO 裡。本機 run 的鍵是解析後的標準形式裝置（`cpu`、`cuda:0`、`mps:0`）；worker run 的鍵是 `<worker>/<要求的裝置>`（`gpu-box-1/cuda`、`any/cpu`）。裝置字串裡不會出現 `/`，所以兩類鍵不會相撞。有 slot 空出時，`RunService.reschedule` 把每條 FIFO 的第一個 run 交給排程器（`core/scheduling.py`），跨所有 FIFO 依送出先後、一次啟動一個；排程器回答一個 `Placement`，或 `None`（該 run 繼續排在第一位）。數條 FIFO 可能共用同一個 slot（`cuda:0`、`any/cuda` 與同機 worker 的 `<worker>/cuda:0`），先服務最早送出的第一位 run，可以避免一條很長的佇列讓其他佇列餓死。取消 FIFO 最前面的 run 時，下一個 run 會立即交給排程器。

**准入以每個 slot 的數量計算**（[#768](https://github.com/CodefyUI/CodefyUI/issues/768)、[#773](https://github.com/CodefyUI/CodefyUI/issues/773)）。加速器的 slot 容納 `CODEFYUI_RUN_QUEUE_MAX_CONCURRENT_GPU` 個 run（1），`cpu` 容納 `CODEFYUI_RUN_QUEUE_MAX_CONCURRENT_CPU` 個（2）。`CODEFYUI_RUN_QUEUE_MAX_CONCURRENT` 覆寫個別 slot：單寫 `cuda:0=2` 指伺服器自己的卡，`gpu-box-1/cuda:0=3` 指該 worker 的卡。目前沒有任何准入判斷讀取可用記憶體；依記憶體准入是 [#788](https://github.com/CodefyUI/CodefyUI/issues/788)。

**裝置集合。** 帶裝置集合的 run 需要集合中每個裝置各一個 slot：全部都有空位才開始，一次全部占用、一次全部釋放。它在 FIFO 最前面等待時，所需的 slot 會被保留，所以其中一張卡上持續的流量無法讓它餓死。保留依先來先到，因此兩個集合 run 不會互相保留而死結。`any` 的集合 run 不保留任何 slot，因為放置之前不知道會在哪台機器。在伺服器本身，集合 run 占用集合中每個 slot，並在第一個裝置上執行；在 worker 上，它的子程序看得到集合中的每個裝置。

**同機共用。** 從 loopback 連入、且 `hello.machine_id` 與伺服器自己相同的 worker（`/etc/machine-id`、macOS 硬體 UUID 或 Windows machine GUID 的 SHA-256 摘要）就在同一台機器上。它的 CPU、`mps:0`，以及 NVML UUID 與伺服器某張卡相符的每張 CUDA 卡，都會對應到伺服器那張卡的 slot，受伺服器的上限約束。因此在上限為 1 時，本機 run 與 worker run 不可能同時拿到同一張實體卡。兩者缺一不可：`ssh -R` 通道會讓遠端 agent 看起來像 loopback，而 machine id 是 agent 自己回報的。

**`target` 選項**（[#769](https://github.com/CodefyUI/CodefyUI/issues/769)）是 `{"worker": "local" | "any" | <名稱>, "devices": [...]}`，兩個鍵都可省略。`worker` 是符合 `[a-z0-9][a-z0-9-]{0,62}` 的名稱；`local` 與 `any` 為保留字。`devices` 列出同一種類、不重複的具體裝置（`cpu`、`cuda:N`、`mps:N`），第一項成為 run 的 `device`。排程器這樣放置：

- **`local`**（或沒有 target）：`LocalScheduler`，即 3.0 的規則。
- **worker 名稱**：放到該 worker，用裝置集合、或要求的裝置、或該 worker 提供的同種類裝置中第一個有空 slot 的。
- **`any`**：依名稱順序第一個已連線、未 drain、能接下這個 run，且不缺少伺服器有的節點型別的 worker（缺少型別或未回報節點目錄的 worker 會被略過；只多出型別的不會），伺服器排最後。CUDA 裝置集合超過一張卡、或是 `cuda:0` 以外的單張卡時，伺服器永遠不符合資格：worker 的 run 子程序看到的集合會從 `cuda:0` 重新編號（`CUDA_VISIBLE_DEVICES`），伺服器的程序則不會，所以固定在 `cuda:1` 的節點會用到另一張卡。其他集合（`cpu`、`mps`、只有 `cuda:0`）在伺服器擁有集合中每個裝置時符合資格。

**為什麼有些拒絕是立即的。** 送出若永遠無法照要求執行，就回 `400`：既未加入、伺服器啟動以來也從未連線過的 worker 名稱、worker（依它最後回報的裝置）不提供要求的裝置、伺服器與它見過的所有 worker 都無法服務的 `any` run（重新編號會改變的 CUDA 集合，則是它見過的所有 worker 都無法服務），worker target 搭配 interactive 通道或 `record_outputs`，或指定的降級 worker 缺少圖（展開 preset 與 subgraph 之後）用到的節點型別。把這種 run 停在佇列裡，會留下一筆永遠不會開始的 `queued` 列，客戶端也無法分辨它與單純很慢的佇列。之後能執行的 run 則會等待，包括目標為已知但目前離線之 worker 的 run；若原因不只是 slot 忙碌，run 列上的 `queue_reason` 會說明：`waiting for worker gpu-box to reconnect`、`worker gpu-box is draining`、`waiting for a worker that offers cuda:0, cuda:1`。drain 一個 worker（`POST /api/workers/{id}/drain`）會停止往它放新的 run；它正在執行的 run 繼續。

**Sweep 與 study**（[#777](https://github.com/CodefyUI/CodefyUI/issues/777)）。`options.target` 會傳到每個子 run。在寫入任何資料列之前，`RunService.check_placeable_batch` 會對每個不同的子 run（子 run 之間只差在圖的 `settings`）套用上面的送出規則，所以 sweep 或 study 要嘛整批被拒絕、要嘛整批被接受；送出子 run 途中有 worker 離線，只會讓它們等待，不會讓整批失敗。只要子 run 還存在，每個 sweep variant 與 study attempt 都會回報 `worker_id`、`devices` 與 `queue_key`。在這台伺服器上帶種子的 run 會占用程序層級的種子閘門而單獨執行；worker run 有自己的隨機狀態，不經過這道閘門，所以在不同 worker 或不同卡上帶種子的子 run 可以同時執行。**新增掃描**對話框提供與 **Queue on...** 相同的目標選擇器，sweep 檢視會顯示每個 variant 的 Worker 與裝置；study 請透過 API 設定 `target`。

## 隔離 {/* #isolation */}

**每個 run 一個子程序**（[#771](https://github.com/CodefyUI/CodefyUI/issues/771)）。每收到一個 `assign`，worker 就在 `multiprocessing` 的 *spawn* context 中啟動 `child.main`，所以每個 run 都有全新的直譯器與自己的 CUDA context。segfault、記憶體不足被系統終止或強制停止，都只結束那一個 run；agent、控制面與其他 run 照常運作。agent 無法履行的指派會立即被拒絕（回一個狀態為 `failed` 的 `run_finished`）：`record_outputs`、它不提供的裝置、它已在執行的 run 的第二份，或 drain 期間的任何 run。

**裝置可見性**（[#774](https://github.com/CodefyUI/CodefyUI/issues/774)）。子程序環境設定 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，讓 agent 透過 NVML 回報的編號就是 torch 使用的編號；並把 `CUDA_VISIBLE_DEVICES` 設為 run 的卡，順序與指派列出的相同。以 `["cuda:2", "cuda:0"]` 為例，run 的 `cuda:0` 就是機器的 `cuda:2`。`cpu` 或 `mps` run 的 `CUDA_VISIBLE_DEVICES` 為空。節點的 `device` 參數從不被改寫，`device_utils.resolve_device` 也不變；重新編號由環境完成。子程序也會移除 `CODEFYUI_DB_PATH`、agent 的 token 與 ROCm 的可見性過濾變數。`run_started` 帶有 `visible_devices` 與 `device_aliases`，控制面會把它們併入 run 的 `execution_facts`。

**取消。** 對 worker run 按 Stop 會轉送為 `cancel`。agent 往 pipe 送 `("cancel",)`，子程序把它轉成 `ExecutionContext.cancel()`，與 in-process run 相同的協作式停止。若子程序在 `grace_s`（預設 30 秒）後仍未結束，agent 會 terminate 它，5 秒後再 kill；控制面把這個 run 記為停止，與 in-process 取消相同。`cdui agent stop` 以同樣方式要求每個執行中的 run 停止，寬限 10 秒。

**斷線。** 關閉 `/ws/worker` 本身從不取消 run。socket 關閉、或連續錯過三次心跳的 worker 會被標為 `disconnected`。agent 送出每個 run 的 frame 前先寫入 `agent-runs/<run id>/spool.jsonl`，控制面確認（`ack`，每記錄 64 個 frame 或 1 秒一次）後即丟棄。每次送出 `ack` 前，控制面會先寫入這個 run 緩衝中的 metrics，並把已確認的 seq 與 run 占用的 slot 存到 `execution_facts.worker_resume`。斷線 worker 的每個 run 變成 `disconnected`（附帶 kind 為 `worker_disconnected` 的 `run_warning`），slot 保留。同名連線若在 `CODEFYUI_WORKER_RESUME_GRACE_S`（300 秒）內於 `run_alive` 列出它，就依序補送未確認的 frame；registry 丟棄已見過的 `seq` 並記錄缺漏，run 回到 `running`（`worker_reconnected`）。逾時則以 `interrupted` 結束並釋放 slot；worker 仍回報但這裡未追蹤的 run 會收到 `grace_s` 為 0 的 `cancel`，它的 `run_finished` 記錄為 `execution_facts.late_result`，不會在結束事件之後再加事件。worker 送來的任何 frame 都算存活訊號，agent 在補送期間也持續送心跳。單一 run 的 inbox 超過 10,000 個 frame 時暫停讀取 socket，讓補送由 TCP 控制速度。重新啟動的控制面會接手前一個程序留下的 `running` 或 `disconnected` worker run（#793）：`recover_interrupted` 照舊把本機與佇列中的資料列標為 interrupted，並把每個 worker 資料列連同存下的 seq 與 slot 交給 `WorkerRegistry.adopt`。資料列變成 `disconnected`，run 重新占用 slot，`WorkerExecutor` 不送 `assign`，只等 worker 回來；尚未認識的 worker（以 session token 連線的 agent）會先有一個不列出的佔位項目，直到它連線。之後就與斷線的 run 相同：寬限內的 `run_alive` 讓它重新接上，補送依存下的 seq 去除重複，所以最後一次 `ack` 之後記錄的 frame 會再收到一次。沒有記錄 `worker_resume` 的資料列（由 3.1.0 啟動）照舊標為 interrupted，因為無法得知它占用的 slot。有 child 被接手的 study 不會在啟動時整理，那個 child 視為仍在執行。正常關閉時，worker 尚未回來的被接手 run 會立即以 `interrupted` 結束。見[連線中斷時](../usage/remote-workers#when-the-connection-drops)。

**留在程序內的部分。** 在控制面內執行的 run 與 3.0 相同，共用一個程序與一個 CUDA context。因此它們的 `peak_memory` 項目標為 `exact: false`；worker run 的 CUDA 項目則是 `exact: true`，因為它的子程序只裝著那一個 run。

## 儲存 {/* #storage */}

**每個 run。** run 子程序寫到節點要寫的位置，`artifact` frame 帶有節點的 `kind`、`path`、`meta` 與 `node_id`；控制面把這一列存進 `exec_run_artifacts`。每個 run 的主控台輸出寫到 agent 使用者資料目錄下的 `agent-runs/<run id>/output.log`，未回報就崩潰時，最後 4 KB 會一併送回。run 子程序從不開啟 SQLite。使用資料夾中檔案的 run（`core/run_needs.py`：上傳欄位、寫檔節點、會存 checkpoint 的 TrainingLoop、資料夾中的路徑）需要一台拿得到這些檔案的 worker，方式有兩種；設計參考 Kubernetes volume 與 Argo 的 artifact repository。

**Mount**（[#782](https://github.com/CodefyUI/CodefyUI/issues/782)、`core/mounts.py`）。worker 為共用資料夾宣告標籤（`cdui agent start --mount data=/nas/...`，以 `hello.mounts` 送出），控制面的資料夾則是標籤 `data`（`CODEFYUI_MOUNT_LABEL`）。`WorkerScheduler` 只把使用檔案的 run 放到擁有該標籤的 worker；指派會讓 run 子程序的資料夾指向 mount，run 記錄的檔案以 `backend: "mount"`、`mount_label` 與相對於 mount 的路徑儲存，讀回時轉換為控制面上的路徑。同機 worker 不必宣告就擁有該標籤：指派會把控制面的資料夾借給它，它的檔案就是一般的 `local` 資料列。

**Transfer**（[#785](https://github.com/CodefyUI/CodefyUI/issues/785)、`core/transfer.py`、`agent/transfer.py`）。適用於沒有該標籤的 worker，且 run 使用的檔案全部是上傳檔、總大小不超過 `CODEFYUI_TRANSFER_MAX_INPUT_BYTES`：`assign.storage` 的 `backend` 為 `"transfer"`，並以 SHA-256 列出輸入檔（存在 `file_hashes` 資料表，檔案大小或 mtime 改變時重新計算）。agent 從下載路徑以 worker token 取得 `agent-cache` 中沒有的檔案，給 run 一個自己的資料夾，並在 `run_finished` 之前，由獨立的執行緒把 run 寫出的檔案上傳到 `POST /api/runs/{id}/artifacts/upload`，中斷 checkpoint 最後上傳。這些資料列的 `backend` 為 `"transfer"`：artifact 放在 `<資料夾>/runs/<run id>/`，run 保留機制與刪除會一併移除；寫檔節點的檔案放在資料夾中的相對路徑，會保留。有 mount 的 worker 優先。

面向使用者的說明，包括拒絕規則與範例，見[遠端 Worker：儲存](/usage/remote-workers#storage)。

## 安全 {/* #security */}

**誰能連線。** `/ws/worker` 只在以下條件全部成立時接受連線，否則在 `accept` 之前關閉：

- `Host` 標頭在允許清單內，與 `/ws/execution` 相同；
- 沒有 `Origin` 標頭（agent 不會送，網頁也沒有理由開這個 socket）；
- 某個驗證器（`ws_worker.py` 的 `AUTHORISERS`）接受 `X-CodefyUI-Token`（或 `?token=`）中的憑證：來自 loopback 對端的伺服器 **session token**（同機的 agent 從伺服器寫出的 token 檔讀取），或來自任何對端的 **worker token**（agent 加入後取得的長期憑證，見[遠端 Worker](/usage/remote-workers#adding-a-worker)）。

被拒絕的握手以 **4003**（Host、Origin）或 **4401**（憑證）關閉。`accept` 之後，第一個 frame 必須在 10 秒內送出 `hello`。它必須滿足的條件由 `core/worker_handshake.py` 決定（[#781](https://github.com/CodefyUI/CodefyUI/issues/781)）：相同的協定版本、相同的 CodefyUI 版本（不同時會被拒絕並列出雙方版本，`version mismatch: worker 3.1.0, control plane 3.1.2`），以及合法且非保留字的名稱；其他情況都會收到 `reject`，原因會依名稱保留給 Workers 面板。身分綁定憑證：worker token 代表它的 worker，所以 `hello` 必須使用該名稱；新連線只有在以相同方式驗證時，才能取代同名的現有連線。

**加入。** Workers 面板的「新增 worker」對話框向 `POST /api/workers/enrollments` 取得一次性 token（`abcdef.0123456789abcdef`，預設 24 小時、可用一次），並顯示 `cdui agent start --control <url> --token <token>`。agent 在 `POST /api/workers/enroll`（不需 session token，依用戶端位址限速）用它換得 worker token，存在 0600 檔案中，每次重啟都沿用。`workers` 與 `worker_enrollments` 兩張表只保存 sha256 雜湊值。已加入的名稱即被占用：只有綁定該 worker 的 token（重新核發）能再次加入，並撤銷舊憑證。`DELETE /api/workers/{id}` 撤銷 worker、中斷連線，並讓佇列中等它的 run 失敗。已加入的 worker 在開機時以 `offline` 載入，所以 run 可以指定它並等待。worker token 或加入請求若從其他機器以明文 HTTP 傳送，會記錄警告。REST 端與 run 路由相同：`GET /api/workers` 開放，`POST /api/workers/{id}/drain` 與 `/undrain` 需要 session token（[API 參考](./api-reference#runs-and-sweeps)）。

**指派中傳送的內容。** `assign` 帶著 run 的圖，其中 SECRET 參數已從記憶體保險箱還原（[#251](https://github.com/CodefyUI/CodefyUI/issues/251)），因為 worker 沒有自己的保險箱。資料庫中的圖仍是去除機密後的版本。對遠端 worker，它會經過網路：在不信任的網路上，請把控制面放在 TLS 之後。

**方向規則。** worker 一律主動連到控制面，從不監聽任何埠；控制面從不連向 worker。worker 從不寫 SQLite。

**節點目錄與指紋**（[#781](https://github.com/CodefyUI/CodefyUI/issues/781)）。agent 永遠不 import torch，所以啟動時由一個短暫的輔助子程序以 run 的方式探索節點，計算節點目錄（`node_registry.node_catalog`：每個節點型別連同 `GET /api/nodes` 提供的定義的 sha256；參數選項採宣告值，不依機器的裝置縮減，所以硬體不影響它）與環境指紋。`hello` 帶著這兩者。節點目錄雜湊與伺服器不同時，worker 會標為 `degraded`，並列出雙方各自缺少的型別；伺服器自己的節點目錄改變時會重新比對。以 worker 為目標的 run，會在送出時於另一個執行緒中計算一次它的節點型別（`worker_handshake.graph_node_types`：準備好的圖，加上 Map 執行的每個已安裝 preset 中的節點）；指定的 worker 若節點目錄不同，就必須擁有全部型別，`WorkerExecutor` 在 `assign` 前也會再檢查一次。指紋文件與 run 的指紋存在同一處（`exec_run_fingerprints`），摘要記在 `workers` 那一列，worker 執行的每個 run 也以它作為 `fingerprint_digest`。

**協定演進。** 兩端執行相同版本，協定仍以新增的方式演進：每個 frame 都會忽略不認得的欄位，`hello`／`welcome` 列出送出方支援的選用功能（`capabilities`，目前為 `ack`、`mounts` 與 `transfer`）；只有雙方清單都列出的功能才會使用，因此只有 `hello` 列出 `transfer` 的 worker 才會收到經 transfer 後端的 run。只有對方會誤讀的變更才會調升 `PROTOCOL_VERSION`（3：`ack` 隨 2 加入，mount 欄位隨 3 加入）；新增的功能改為宣告 capability。

**TLS。** Agent 會以這台機器的 CA 或釘選的金鑰驗證控制面的 TLS 憑證，機密應只經由已驗證的通道傳送：見 [遠端 Worker，agent 與控制面之間的信任](/usage/remote-workers#trust-between-agent-and-control-plane)（[#786](https://github.com/CodefyUI/CodefyUI/issues/786)）。

## 不變量 {/* #invariants */}

這個領域的每項變更都要遵守以下規則（[#764](https://github.com/CodefyUI/CodefyUI/issues/764)）：

1. 沒有 worker 註冊、也沒有 `target` 選項時，送出的行為與 3.0.2 完全相同：同一個 queue key、同一個 `InProcessExecutor`、同一組事件序列。`single_node_parity` 快照只在有意變更它的 commit 中改變。
2. **匯出面凍結。** `backend/app/core/codegen.py`、`backend/run_graph.py`、節點 `execute()` 的語意，以及匯出腳本 import 的執行期符號：`graph_engine.invoke_node`、`prepare_executable_graph` 與 `topological_sort`；`ExecutionContext` 建構子與 `derive_seed`、`current_node_id`；`device_utils.resolve_device`；`seeding.*`、`runtime.*` 與 `api_contract.*`。run 子程序會建構 `ExecutionContext`，但從不改變它的介面；裝置集合以 `CUDA_VISIBLE_DEVICES` 實作，`resolve_device` 維持原樣。
3. Interactive（畫布）通道一律在控制面程序內執行。
4. worker 從不開啟入站埠，也從不寫 SQLite。
5. 新行為都在 `target` 選項或 `CODEFYUI_EXECUTION_*` 設定之後，預設關閉。
6. 網頁優先：worker 在瀏覽器中管理，CLI 只新增 `cdui agent start|stop`。不新增 `cdui status` 或 TUI 功能。

## 決策紀錄 {/* #decisions-log */}

| 日期 | 決策 |
|------|----------|
| 2026-10-15 | 把執行從控制面拆出；保留 in-process 路徑作為預設，也是唯一的單節點路徑。 |
| 2026-10-15 | 網頁優先：Workers 面板、「新增 worker」對話框與 run 目標選擇器都在瀏覽器中；CLI 只保留啟動、加入與停止。 |
| 2026-10-15 | 拓撲：任何機器都可以 `cdui start` 作為控制面；GPU 機器執行 `cdui agent start`，以瀏覽器產生的一行指令加入。連線方向一律是 worker → 控制面。 |
| 2026-10-15 | 准入維持以每個 `(worker, device)` slot 計數；依記憶體准入延後（[#788](https://github.com/CodefyUI/CodefyUI/issues/788)）。 |
| 2026-10-15 | 儲存：預設使用 NAS（mount）；也必須有不需 NAS 的 transfer 後端。 |
| 2026-10-15 | Interactive 通道留在控制面；worker 憑證透過 enrollment 取得；先支援 Linux worker，Windows 盡力而為；控制面與 worker 版本須相同；DDP 之後再做（[#792](https://github.com/CodefyUI/CodefyUI/issues/792)）。 |
| 2026-10-15 | 每個 run 一個 spawn 子程序，裝置集合以 `CUDA_VISIBLE_DEVICES` 傳入（[#771](https://github.com/CodefyUI/CodefyUI/issues/771)）；agent 程序從不 import torch。 |
| 2026-10-15 | 同機 worker 以 machine id 判定，共用的卡以 NVML UUID 判定，從不依對端位址（[#773](https://github.com/CodefyUI/CodefyUI/issues/773)）。 |
| 2026-10-15 | enrollment 完成之前，`/ws/worker` 只接受帶 session token 的 loopback 對端（[#772](https://github.com/CodefyUI/CodefyUI/issues/772)）。 |
| 2026-10-15 | 加入：一次性 token 換成每個 worker 獨立的憑證；session token 仍只限 loopback；身分綁定憑證，已加入的名稱只能透過重新核發 token 易手，同機需要 loopback 對端且 machine id 相符（[#780](https://github.com/CodefyUI/CodefyUI/issues/780)）。 |
| 2026-10-15 | 單一 run、sweep 與 study 共用同一條送出規則：只有啟動以來從未見過的 worker 名稱會被拒絕；目標為已知但離線之 worker 的 run 會等它回來（[#777](https://github.com/CodefyUI/CodefyUI/issues/777)）。 |

## 進入點 {/* #entry-points */}

| 區域 | 檔案 |
|------|------|
| Run service（佇列、run 列、事件紀錄、取消） | `backend/app/core/run_service.py` |
| Executor 接縫、`InProcessExecutor`、`WorkerExecutor` | `backend/app/core/executors.py` |
| 純函式事件建構器 | `backend/app/core/run_events.py` |
| 排程器、slot、placement | `backend/app/core/scheduling.py` |
| Worker registry | `backend/app/core/workers.py` |
| Worker 傳輸協定 | `backend/app/core/worker_protocol.py` |
| 裝置清單（NVML、無 torch 模式） | `backend/app/core/resources.py` |
| 機器識別 | `backend/app/core/machine_identity.py` |
| `/ws/worker` 處理器 | `backend/app/api/ws_worker.py` |
| `/api/workers` 路由 | `backend/app/api/routes_workers.py` |
| Worker agent（CLI、連線、runner、run 子程序） | `backend/app/agent/` |
| Workers 面板 | `frontend/src/components/Workers/` |
| Queue on... 對話框 | `frontend/src/components/Toolbar/QueueRunDialog.tsx`、`frontend/src/components/Toolbar/queueTargets.ts` |
