---
sidebar_position: 6.5
title: 執行架構
description: run 可以在伺服器程序內或 worker agent 上執行時，從送出到結果的路徑——控制面、executor、worker agent 與 run 子程序、slot 與 placement、事件路徑、隔離、儲存、安全，以及每項變更都要遵守的不變量。
---

# 執行架構

[架構](./architecture)說明整個伺服器。本頁深入其中一部分：run 的圖在哪裡執行。預設情況下，每個 run 都在伺服器程序內執行，與 3.0 完全相同。queued run 也可以改送到 **worker agent**（`cdui agent start`），由它在自己的子程序中執行。設計追蹤於 [#764](https://github.com/CodefyUI/CodefyUI/issues/764)；使用方式見[執行佇列：Workers](/usage/run-queue#workers) 與 [CLI 指令：Agent 指令](/getting-started/cli-commands#agent-commands)。

:::note 現況
目前程式碼已有：executor 接縫、依 `(worker, device)` slot 放置、`target` 選項（單一 run、sweep 與 study 皆可用）、與伺服器在**同一台機器**上的 worker agent、裝置集合、每個 run 的峰值記憶體、Workers 面板，以及工具列的 **Queue on...** 對話框與 Runs 面板的 Worker、Devices 欄。已規劃但尚未進入程式碼：enrollment token 與遠端 worker（[#780](https://github.com/CodefyUI/CodefyUI/issues/780)）、握手時檢查版本與節點目錄（[#781](https://github.com/CodefyUI/CodefyUI/issues/781)）、儲存 mount（[#782](https://github.com/CodefyUI/CodefyUI/issues/782)）與 transfer 後端（[#785](https://github.com/CodefyUI/CodefyUI/issues/785)）、讓 run 撐過斷線的 spool（[#783](https://github.com/CodefyUI/CodefyUI/issues/783)），以及控制面指紋釘選（[#786](https://github.com/CodefyUI/CodefyUI/issues/786)）。以下各節會註明哪些部分仍在規劃中。
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
| **Mount**、**transfer 後端** | 規劃中的儲存名詞（[儲存](#storage)）。 |

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
| agent → 控制面 | `hello`（名稱、版本、協定版本、環境指紋、節點目錄雜湊、machine id、裝置、mount、平台）、`heartbeat`（含可用記憶體的裝置、執行中的 run id）、`run_alive`（重新連線後），以及每個 run 的 `run_started`、`event`、`metrics`、`artifact`、`warning`、`run_finished` |
| 控制面 → agent | `welcome`（worker id、`heartbeat_s`、`grace_s`）、`reject`（原因，然後關閉）、`assign`（run id、圖、選項、device、devices、選用的 `grace_s`）、`cancel`（run id、選用的 `grace_s`）、`drain` |

## 排程 {/* #scheduling */}

**Queue key 與 slot。** 每個等待中的 run 都在一條以字串為鍵的 FIFO 裡。本機 run 的鍵是解析後的標準形式裝置（`cpu`、`cuda:0`、`mps:0`）；worker run 的鍵是 `<worker>/<要求的裝置>`（`gpu-box-1/cuda`、`any/cpu`）。裝置字串裡不會出現 `/`，所以兩類鍵不會相撞。有 slot 空出時，`RunService.reschedule` 把每條 FIFO 的第一個 run 交給排程器（`core/scheduling.py`），跨所有 FIFO 依送出先後、一次啟動一個；排程器回答一個 `Placement`，或 `None`（該 run 繼續排在第一位）。數條 FIFO 可能共用同一個 slot（`cuda:0`、`any/cuda` 與同機 worker 的 `<worker>/cuda:0`），先服務最早送出的第一位 run，可以避免一條很長的佇列讓其他佇列餓死。取消 FIFO 最前面的 run 時，下一個 run 會立即交給排程器。

**准入以每個 slot 的數量計算**（[#768](https://github.com/CodefyUI/CodefyUI/issues/768)、[#773](https://github.com/CodefyUI/CodefyUI/issues/773)）。加速器的 slot 容納 `CODEFYUI_RUN_QUEUE_MAX_CONCURRENT_GPU` 個 run（1），`cpu` 容納 `CODEFYUI_RUN_QUEUE_MAX_CONCURRENT_CPU` 個（2）。`CODEFYUI_RUN_QUEUE_MAX_CONCURRENT` 覆寫個別 slot：單寫 `cuda:0=2` 指伺服器自己的卡，`gpu-box-1/cuda:0=3` 指該 worker 的卡。目前沒有任何准入判斷讀取可用記憶體；依記憶體准入是 [#788](https://github.com/CodefyUI/CodefyUI/issues/788)。

**裝置集合。** 帶裝置集合的 run 需要集合中每個裝置各一個 slot：全部都有空位才開始，一次全部占用、一次全部釋放。它在 FIFO 最前面等待時，所需的 slot 會被保留，所以其中一張卡上持續的流量無法讓它餓死。保留依先來先到，因此兩個集合 run 不會互相保留而死結。`any` 的集合 run 不保留任何 slot，因為放置之前不知道會在哪台機器。在伺服器本身，集合 run 占用集合中每個 slot，並在第一個裝置上執行；在 worker 上，它的子程序看得到集合中的每個裝置。

**同機共用。** `hello.machine_id` 與伺服器自己相同的 worker（`/etc/machine-id`、macOS 硬體 UUID 或 Windows machine GUID 的 SHA-256 摘要）就在同一台機器上。它的 CPU、`mps:0`，以及 NVML UUID 與伺服器某張卡相符的每張 CUDA 卡，都會對應到伺服器那張卡的 slot，受伺服器的上限約束。因此在上限為 1 時，本機 run 與 worker run 不可能同時拿到同一張實體卡。這無法由對端位址判斷：`ssh -R` 通道會讓遠端 agent 看起來像 loopback。

**`target` 選項**（[#769](https://github.com/CodefyUI/CodefyUI/issues/769)）是 `{"worker": "local" | "any" | <名稱>, "devices": [...]}`，兩個鍵都可省略。`worker` 是符合 `[a-z0-9][a-z0-9-]{0,62}` 的名稱；`local` 與 `any` 為保留字。`devices` 列出同一種類、不重複的具體裝置（`cpu`、`cuda:N`、`mps:N`），第一項成為 run 的 `device`。排程器這樣放置：

- **`local`**（或沒有 target）：`LocalScheduler`，即 3.0 的規則。
- **worker 名稱**：放到該 worker，用裝置集合、或要求的裝置、或該 worker 提供的同種類裝置中第一個有空 slot 的。
- **`any`**：依名稱順序第一個已連線、未 drain 且能接下這個 run 的 worker，伺服器排最後。帶裝置集合時，伺服器永遠不符合資格：worker 的 run 子程序看到的集合會從 `cuda:0` 重新編號（`CUDA_VISIBLE_DEVICES`），伺服器的程序則不會，所以固定在 `cuda:1` 的節點會用到另一張卡。

**為什麼有些拒絕是立即的。** 送出若永遠無法照要求執行，就回 `400`：伺服器啟動以來從未連線過的 worker 名稱、worker（依它最後回報的裝置）不提供要求的裝置、伺服器與它見過的所有 worker 都無法服務的 `any` run（裝置集合則是它見過的所有 worker 都無法服務），或 worker target 搭配 interactive 通道或 `record_outputs`。把這種 run 停在佇列裡，會留下一筆永遠不會開始的 `queued` 列，客戶端也無法分辨它與單純很慢的佇列。之後能執行的 run 則會等待，包括目標為已知但目前離線之 worker 的 run；若原因不只是 slot 忙碌，run 列上的 `queue_reason` 會說明：`waiting for worker gpu-box to reconnect`、`worker gpu-box is draining`、`waiting for a worker that offers cuda:0, cuda:1`。drain 一個 worker（`POST /api/workers/{id}/drain`）會停止往它放新的 run；它正在執行的 run 繼續。

**Sweep 與 study**（[#777](https://github.com/CodefyUI/CodefyUI/issues/777)）。`options.target` 會傳到每個子 run。在寫入任何資料列之前，`RunService.check_placeable_batch` 會對每個不同的子 run（子 run 之間只差在圖的 `settings`）套用上面的送出規則，所以 sweep 或 study 要嘛整批被拒絕、要嘛整批被接受；送出子 run 途中有 worker 離線，只會讓它們等待，不會讓整批失敗。只要子 run 還存在，每個 sweep variant 與 study attempt 都會回報 `worker_id`、`devices` 與 `queue_key`。在這台伺服器上帶種子的 run 會占用程序層級的種子閘門而單獨執行；worker run 有自己的隨機狀態，不經過這道閘門，所以在不同 worker 或不同卡上帶種子的子 run 可以同時執行。**新增掃描**對話框提供與 **Queue on...** 相同的目標選擇器，sweep 檢視會顯示每個 variant 的 Worker 與裝置；study 請透過 API 設定 `target`。

## 隔離 {/* #isolation */}

**每個 run 一個子程序**（[#771](https://github.com/CodefyUI/CodefyUI/issues/771)）。每收到一個 `assign`，worker 就在 `multiprocessing` 的 *spawn* context 中啟動 `child.main`，所以每個 run 都有全新的直譯器與自己的 CUDA context。segfault、記憶體不足被系統終止或強制停止，都只結束那一個 run；agent、控制面與其他 run 照常運作。agent 無法履行的指派會立即被拒絕（回一個狀態為 `failed` 的 `run_finished`）：`record_outputs`、它不提供的裝置、它已在執行的 run 的第二份，或 drain 期間的任何 run。

**裝置可見性**（[#774](https://github.com/CodefyUI/CodefyUI/issues/774)）。子程序環境設定 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，讓 agent 透過 NVML 回報的編號就是 torch 使用的編號；並把 `CUDA_VISIBLE_DEVICES` 設為 run 的卡，順序與指派列出的相同。以 `["cuda:2", "cuda:0"]` 為例，run 的 `cuda:0` 就是機器的 `cuda:2`。`cpu` 或 `mps` run 的 `CUDA_VISIBLE_DEVICES` 為空。節點的 `device` 參數從不被改寫，`device_utils.resolve_device` 也不變；重新編號由環境完成。子程序也會移除 `CODEFYUI_DB_PATH`、agent 的 token 與 ROCm 的可見性過濾變數。`run_started` 帶有 `visible_devices` 與 `device_aliases`，控制面會把它們併入 run 的 `execution_facts`。

**取消。** 對 worker run 按 Stop 會轉送為 `cancel`。agent 往 pipe 送 `("cancel",)`，子程序把它轉成 `ExecutionContext.cancel()`，與 in-process run 相同的協作式停止。若子程序在 `grace_s`（預設 30 秒）後仍未結束，agent 會 terminate 它，5 秒後再 kill；控制面把這個 run 記為停止，與 in-process 取消相同。`cdui agent stop` 以同樣方式要求每個執行中的 run 停止，寬限 10 秒。

**斷線。** 關閉 `/ws/worker` 本身從不取消 run。socket 關閉、或連續錯過三次心跳的 worker 會被標為 `disconnected`；它正在執行的每個 run 以 `interrupted` 結束，附帶一個 kind 為 `worker_disconnected` 的 `run_warning`，其 slot 也會釋放。同名的第二條連線會取代第一條；舊連線上仍在進行的 run，若新連線在 `run_alive` 中列出就繼續，否則以 `interrupted` 結束。讓 run 撐過較長的中斷是 [#783](https://github.com/CodefyUI/CodefyUI/issues/783)。

**留在程序內的部分。** 在控制面內執行的 run 與 3.0 相同，共用一個程序與一個 CUDA context。因此它們的 `peak_memory` 項目標為 `exact: false`；worker run 的 CUDA 項目則是 `exact: true`，因為它的子程序只裝著那一個 run。

## 儲存 {/* #storage */}

**目前。** worker 與伺服器在同一台機器上，兩者看到同一個檔案系統。run 子程序寫到節點要寫的位置，`artifact` frame 帶有節點的 `kind`、`path`、`meta` 與 `node_id`；控制面把這一列存進 `exec_run_artifacts`，路徑與 worker 回報的完全相同。每個 run 的主控台輸出寫到 agent 使用者資料目錄下的 `agent-runs/<run id>/output.log`，未回報就崩潰時，最後 4 KB 會一併送回。run 子程序從不開啟 SQLite。

**規劃中。** 遠端 worker 需要能找到 run 的輸入檔，並把 run 寫出的東西送回。設計參考 Kubernetes volume 與 Argo 的 artifact repository：

- **Mount**（[#782](https://github.com/CodefyUI/CodefyUI/issues/782)）：worker 為共享檔案系統宣告標籤（`data` → `/mnt/nas/data`），控制面在自己與 worker 的路徑之間轉換，排程器只把 run 放到擁有所需標籤的 worker。`hello` 與 `GET /api/workers` 中的 `mounts` 欄位已為此保留。
- **Transfer 後端**（[#785](https://github.com/CodefyUI/CodefyUI/issues/785)）：沒有共享檔案系統時，由控制面擔任 artifact repository，檔案以內容定址的方式透過 HTTP 搬運。

## 安全 {/* #security */}

**目前：僅限同機。** `/ws/worker` 只在以下條件全部成立時接受連線，否則在 `accept` 之前關閉：

- `Host` 標頭在允許清單內，與 `/ws/execution` 相同；
- 沒有 `Origin` 標頭（agent 不會送，網頁也沒有理由開這個 socket）；
- 對端位址是 loopback；
- 請求帶有伺服器的 session token（`?token=` 或 `X-CodefyUI-Token`），同機的 agent 從伺服器寫出的 token 檔讀取。

被拒絕的握手以 **4003**（Host、Origin、非 loopback 對端）或 **4401**（token）關閉。`accept` 之後，第一個 frame 必須在 10 秒內送出 `hello`；協定版本不符、名稱不合法或為保留字，會收到 `reject`。REST 端與 run 路由相同：`GET /api/workers` 開放，`POST /api/workers/{id}/drain` 與 `/undrain` 需要 session token（[API 參考](./api-reference#runs-and-sweeps)）。

**指派中傳送的內容。** `assign` 帶著 run 的圖，其中 SECRET 參數已從記憶體保險箱還原（[#251](https://github.com/CodefyUI/CodefyUI/issues/251)），因為 worker 沒有自己的保險箱。資料庫中的圖仍是去除機密後的版本。目前這個 frame 只經過 loopback socket。

**方向規則。** worker 一律主動連到控制面，從不監聽任何埠；控制面從不連向 worker。worker 從不寫 SQLite。

**規劃中。** 透過 Workers 面板的「新增 worker」對話框發出 enrollment token，給每個 worker 獨立憑證（[#780](https://github.com/CodefyUI/CodefyUI/issues/780)）；握手時檢查版本、環境指紋與節點目錄（[#781](https://github.com/CodefyUI/CodefyUI/issues/781)）；agent 在首次使用時釘選控制面的 TLS 指紋，並規定機密只能經由已驗證的通道傳送（[#786](https://github.com/CodefyUI/CodefyUI/issues/786)）。

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
