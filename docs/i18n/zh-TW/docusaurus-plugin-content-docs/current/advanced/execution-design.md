---
sidebar_position: 6.6
title: 執行設計筆記
description: 伺服器可以把 run 交給 worker agent 時，run 為何以現在的方式執行——目標、六條不變量與各自的理由、各項決策與被排除的替代方案、依日期排列的決策紀錄，以及仍待決定的問題。
---

# 執行設計筆記

[執行架構](./execution-architecture)說明 run 如何在伺服器程序內或 worker agent 上執行。本頁記錄它為何這樣設計：控制面／worker 分離（[#764](https://github.com/CodefyUI/CodefyUI/issues/764)）的目標、每項變更都要遵守的不變量、各項決策與它們排除的替代方案，以及仍待決定的問題。每項決策都連到記載其行為的頁面；操作說明在[多 GPU 執行](/usage/multi-gpu)、[遠端 Worker](/usage/remote-workers)、[執行佇列](/usage/run-queue)與[實驗研究（Studies）](/usage/studies)。

## 目標 {/* #goals */}

設計由三種情境驅動：

| 情境 | 需求 |
| --- | --- |
| 遠端執行 | 筆電或伺服器上的 `cdui start` 是控制面；每台 GPU 伺服器執行 `cdui agent start`；run 在網頁介面中選擇目標。 |
| 一台機器、多張 GPU | run 選擇一組卡；多個 run 可共用一張卡；放不下的 run 在開始前就被拒絕並附上原因；一個 run 不會弄壞另一個。 |
| 一台機器、一個程序 | 預設安裝、畫布與匯出的 Python 腳本，行為與分離之前完全相同。 |

不在範圍內：使用者帳號與多租戶（後端沒有使用者的概念，[#247](https://github.com/CodefyUI/CodefyUI/issues/247)）、跨機器訓練，以及分片（FSDP）或模型平行訓練。

## 不變量 {/* #invariants */}

這個領域的每項變更都要遵守以下規則。做不到的變更會先停下來，單獨審查。

1. **沒有 worker 註冊、也沒有 `target` 選項時，送出的行為與 3.0.2 完全相同**：同一個 queue key、同一個 `InProcessExecutor`、同一組事件序列。*理由：*單機路徑是大多數安裝與所有課堂使用的路徑，也是 12 月考試執行的路徑。`single-node-parity` CI job 會把預設送出的事件序列與儲存的快照比對，快照只在有意變更它的 commit 中改變。
2. **匯出面凍結**：`backend/app/core/codegen.py`、`backend/run_graph.py`、節點 `execute()` 的語意，以及匯出腳本 import 的執行期符號（`graph_engine.invoke_node`、`prepare_executable_graph`、`topological_sort`；`ExecutionContext` 建構子與 `derive_seed`、`current_node_id`；`device_utils.resolve_device`；`seeding.*`、`runtime.*`、`api_contract.*`）。*理由：*匯出的腳本在執行時從已安裝的 backend import 這些符號，並由線上評測系統檢查輸出，因此任何一項改變都會改變評分結果。run 子程序會建構 `ExecutionContext`，但不改變它的介面；裝置集合以 `CUDA_VISIBLE_DEVICES` 實作，`resolve_device` 因此維持原樣。
3. **Interactive（畫布）通道一律在控制面程序內執行。** *理由：*畫布 run 使用只存在於該程序的狀態：socket 的執行快取、持久化的模組權重，以及 Inspector 讀取的輸出（[決策：畫布留在本機](#interactive-lane-local)）。
4. **worker 從不開啟入站埠，也從不寫 SQLite。** *理由：*GPU 伺服器常位於 NAT 與防火牆之後，會監聽的 worker 需要自己的認證；資料庫只有一個寫入者，也就是控制面（[決策：worker 主動連出](#workers-dial-out)）。
5. **新行為預設關閉，或需要明確的選項才會啟用。** *理由：*只有新功能都不會自行開啟，不變量 1 才成立。程式碼中的開關是 `target` 選項、`provenance` run 選項、卡片的**依記憶體共用**政策（卡片預設為**獨占**）、`TrainingLoop` 的 `strategy`（預設 `single`）、`CODEFYUI_S3_ENABLED`（預設關閉），以及 agent 自己的旗標（`--one-shot`、`--mount`）。
6. **網頁優先。** worker 在瀏覽器中管理；CLI 只新增 `cdui agent start`、`stop` 與 `forget`，不新增 `cdui status` 或 TUI 功能。*理由：*伺服器支援的每個平台上，瀏覽器的表現都一致，終端介面則難以在各平台間維持相容。

## 決策 {/* #decisions */}

### 控制面留在伺服器程序內 {/* #control-plane-in-the-server */}

**決策。** `cdui start` 程序保有佇列、排程器、run 資料列、事件日誌與 worker registry。只有 queued run 的執行移到外面，而且只經過一個接縫：`RunService._execute` 把 `ExecutionRequest` 交給 `Executor`（`InProcessExecutor` 或 `WorkerExecutor`），兩者都透過同一組 sink 寫進同一份事件日誌。

**理由。** run service 原本就擁有每個 run（送出、持久化事件、取消、重啟後復原），它的事件詞彙本來就是線上格式，因此 worker 的事件可以走 in-process run 的同一條路徑。瀏覽器與 `cdui run` 都不需要修改。

**排除的方案。** 搭配訊息代理的獨立排程服務（例如 n8n queue mode 使用的 Redis），或以 Ray、Celery 作為執行後端：每一種都多出一個要安裝與維運的服務，Ray 還與 torch 版本綁定，對 Windows 的支援也有限。多個 uvicorn worker（`cdui start -- --workers N`）：每一個都會是同一個資料庫上的第二台伺服器，instance lock 會拒絕（[#529](https://github.com/CodefyUI/CodefyUI/issues/529)）。規模是幾台 GPU 伺服器、幾十個 run，一個 SQLite 寫入者加一個 event loop 就足以應付。

**記載於**[執行架構：角色](./execution-architecture#roles)與[事件路徑](./execution-architecture#event-path)。

### 畫布留在本機 {/* #interactive-lane-local */}

**決策。** 畫布上的**執行**在控制面程序內執行，從不放到 worker 上；同時指定 interactive 通道與 worker 目標的送出會被拒絕。要把分頁的圖送到 worker，請用 **Queue on...**。

**理由。** 畫布在兩次點擊之間會重用 socket 的執行快取與持久化權重，並為 Inspector 錄製輸出，這些都保存在該程序中。queued run 不使用其中任何一項，因此移到 worker 不會失去什麼。worker run 的錄製輸出是之後才加入的（[#789](https://github.com/CodefyUI/CodefyUI/issues/789)）：它們留在 worker 上，經由 worker 的連線讀取。

**排除的方案。** 把畫布 run 送到 worker：快取與權重儲存必須搬到 worker 上，並能從畫布定址，對畫布用來跑的短 run 卻沒有好處。

**記載於**[執行架構：通道與執行位置](./execution-architecture#lanes-and-where-they-execute)與[多 GPU 執行：畫布的行為](/usage/multi-gpu#the-canvas)。

### worker 主動連出 {/* #workers-dial-out */}

**決策。** agent 開啟通往控制面的 `/ws/worker` 並保持連線；指派、取消、drain 與輸出讀取都經由這條連線回傳。控制面從不連線到 worker。無法直接連到控制面的機器，由控制面啟動的 SSH session 服務（`ssh -R` 轉送到遠端主機上的 Unix socket）。run 子程序永遠拿不到 `CODEFYUI_DB_PATH`，只有控制面寫入資料庫。

**理由。** 實驗室的 GPU 伺服器常在 NAT 或只開 SSH 的防火牆之後，控制面則只需要一個可以連到的 URL。worker 沿用控制面的 Host 檢查與認證，同機 worker 與他處的 worker 也使用同一套程式碼。

**排除的方案。** Push 模式，也就是由控制面呼叫每個 worker 上的 HTTP API（ComfyUI-Distributed、ComfyUI_NetDist）：每個 worker 都要開埠並認證呼叫者，控制面也要知道每一個位址。在每個 worker 的環境變數中放共用密鑰：改以 enrollment 為每台機器核發憑證，才能單獨撤銷其中一台。

**記載於**[執行架構：安全](./execution-architecture#security)、[遠端 Worker：連到控制面](/usage/remote-workers#reaching-the-control-plane)與[新增 worker](/usage/remote-workers#adding-a-worker)。

### 兩端版本相同 {/* #same-version */}

**決策。** CodefyUI 版本不同的 worker 會在握手時被拒絕，原因中列出兩邊的版本。節點目錄不同的 worker 為**降級（Degraded）**：它只接受所用節點型別它都有的圖，而它缺少控制面擁有的型別時，`any` 會跳過它。

**理由。** 兩端共用一個協定模組與一套事件詞彙，run 的圖在兩端也必須代表同一件事。拒絕不同版本比協商版本更容易推論；協定仍以新增的方式成長，搭配 capability，讓選用功能可以逐個 agent 檢查。

**排除的方案。** 在協定協商之下混用不同版本。把控制面的 plugin 複製到 worker 的指令尚未實作（[待決問題](#open-questions)）。

**記載於**[遠端 Worker：讓 worker 保持同步](/usage/remote-workers#keeping-workers-in-sync)與[執行架構：安全](./execution-architecture#security)。

### 每個 run 一個子程序 {/* #one-subprocess-per-run */}

**決策。** agent 以 `multiprocessing` 的 *spawn* context 為每個 run 啟動全新的直譯器。在 worker 上只有這個子程序會 import PyTorch；它的卡以 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 與 `CUDA_VISIBLE_DEVICES` 設定，依 run 列出的順序從 `cuda:0` 重新編號。agent 本身不經 torch，透過 NVML 讀取裝置資訊。

**理由。** 在伺服器程序內，每個 run 共用一個 Python 程序與一個 CUDA context，一次 segfault、一次記憶體不足被系統終止，或一個損毀的 context，都會讓所有 run 與 UI 一起結束。子程序把影響限制在一個 run：寬限期過後可以強制終止（終止執行緒則可能讓裝置無法使用）、每個 run 有自己的亂數狀態，因此不同卡上有種子的 run 可以同時執行，記憶體上限也能套用到單一 run。Windows 上以及 CUDA 初始化之後都必須使用 *spawn*。agent 自己從不持有 CUDA context。

**排除的方案。** 在同一個程序內以執行緒執行，這仍是伺服器執行自己的 run 的方式，也是伺服器上的 run 共用一道種子閘門、峰值記憶體標為 `exact: false` 的原因。每個 run 一個容器：隔離更強，但要設定 GPU passthrough，也沒有 Windows 的做法。預先啟動的直譯器池沒有實作：每個 run 自行負擔 import torch 的時間。

**記載於**[執行架構：隔離](./execution-architecture#isolation)。

### 准入：每個 slot 計數，共用的卡依記憶體預算 {/* #memory-admission */}

**決策。** slot 是一台機器上的一個裝置，`(worker, device)`，同時容納固定數量的 run（加速器 1 個、CPU 2 個，可逐 slot 覆寫）。設為**依記憶體共用**的 worker CUDA 卡則改依預算准入：run 宣告 `target.memory_gib`（或取得同一張圖在相同卡數上最近一次量到的峰值加 20%），當預算加上 `CODEFYUI_EXECUTION_CUDA_CONTEXT_MIB` 不超過 `min(free - reserved, total - reserved - running)` 時就放得下，它的子程序在回報 `run_started` 之前把配置器上限設為預算（`torch.cuda.set_per_process_memory_fraction`）。指定 worker 卻放不下的 run 會立即被拒絕並附上數字，除非 `target.wait` 為 true；`any` run 則會等待。同一個實體裝置的多個 slot（依 NVML UUID 比對，或同一台機器的 CPU 與 `mps:0`）算作一份額度，取各提供者中最小的上限（[#842](https://github.com/CodefyUI/CodefyUI/issues/842)）。

**理由。** 「多個 run 可共用一張卡；放不下的 run 會被拒絕」需要在 run 開始之前做出判斷，而上限讓共用變得安全：超出預算的 run 只會自己記憶體不足。帳本計算的是預算，因為心跳看不到每個 run 的實際用量。headroom 的第二項確保承諾不超過整張卡，因為剛開始的 run 還沒配置多少記憶體。預設仍是計數，因為這是 3.0 的行為，也不需要使用者提供預算。

**排除的方案。** 在伺服器程序內依記憶體准入：那裡的 run 共用一個程序，無法把其中一個限制在預算內，因此 `target.memory_gib` 搭配 `local` 會被拒絕。以圖的參數推算的靜態估計作為伺服器的預設值：Queue on... 對話框會提供它並標示為估計，伺服器自行套用的只有實測峰值。只以 `free - reserved` 准入，也就是 [#788](https://github.com/CodefyUI/CodefyUI/issues/788) 最初的規格：在兩個尚未配置記憶體的 run 旁邊，它會再放進第三個。

**記載於**[多 GPU 執行：Slot 與上限](/usage/multi-gpu#slots-and-limits)、[共用一張卡](/usage/multi-gpu#sharing-a-card)與[執行架構：排程](./execution-architecture#scheduling)。

### 儲存：先 mount，再 transfer {/* #storage */}

**決策。** 使用資料夾中檔案的 run 會送到持有相同 mount 標籤（`data`）的 worker，路徑依各機器轉換；沒有這樣的 worker 時，若它用到的檔案都是上傳檔且在 transfer 上限內，由控制面擔任檔案庫（transfer 後端），或在有設定時改由 S3／MinIO bucket 擔任。兩者都不符合的 run，指定 worker 時會被拒絕，`any` 則跳過該 worker。

**理由。** 實驗室通常有 NAS，共用資料夾不需搬運任何位元組。沒有 NAS 的機器也必須能用。模型取自 Kubernetes 的 volume（pod 被放到它的 volume 所在之處）與 Argo 的 artifact repository（步驟之間透過儲存庫交換檔案，常見的是 MinIO）。

**排除的方案。** 對每個 worker 都經由控制面搬運所有檔案：即使有 NAS，控制面的網路連線也會成為瓶頸。把 artifact 資料列當作單純的路徑：那會指向另一台機器上的檔案；資料列改為記錄後端、mount 標籤與相對路徑。

**記載於**[遠端 Worker：儲存](/usage/remote-workers#storage)與[執行架構：儲存](./execution-architecture#storage)。

### 斷線時保留 run {/* #disconnects */}

**決策。** 連線關閉或連續錯過三次心跳時，worker 與它的 run 會標為 `disconnected`。agent 會把每個 frame 暫存到控制面確認為止，在 `CODEFYUI_WORKER_RESUME_GRACE_S`（300 秒）內重新連線就重送其餘部分；控制面若在沒有停止 run 的情況下重啟，會收養它留下的 worker run（[#793](https://github.com/CodefyUI/CodefyUI/issues/793)）。

**理由。** 作為控制面的筆電會休眠、會換網路，而 worker 上的 run 子程序並不依賴這條 socket。寬限時間參考 Kubernetes 的 node controller，它在節點失聯後會等候數分鐘才驅逐。

**排除的方案。** 連線一斷就取消 run，這會讓每一次網路抖動都賠上整個 run。

**記載於**[遠端 Worker：連線中斷時](/usage/remote-workers#when-the-connection-drops)與[執行架構：隔離](./execution-architecture#isolation)。

### 給批次系統用的一次性 worker {/* #one-shot-workers */}

**決策。** `cdui agent start --one-shot` 只接一個 run，回報後就以表示 run 結果的結束碼退出。`any` 從不把 run 放到一次性 worker 上；run 要指名它，或指定一次性 worker 池（`target.pool: "one-shot"`），才會送到它那裡，並最多等待 `CODEFYUI_WORKER_ONE_SHOT_QUEUE_TIMEOUT_S`（24 小時）。`--instance` 讓多個 job（例如 SLURM array 的各個 task）共用一份憑證。

**理由。** 在叢集上，由批次系統決定 job 拿到哪台機器、哪張 GPU、能用多久。在 job 內啟動的一次性 agent 把這個決定交給 SLURM，CodefyUI 則負責送出圖並記錄結果。做法仿照以 `--ephemeral` 啟動的 GitHub runner。

**排除的方案。** 由控制面送出批次 job（以 SLURM 作為執行後端）：CodefyUI 就必須為每個叢集的 partition、帳號與限制建模。把 `any` run 放到一次性 agent 上：跑完一個 run 就退出的 agent 會拿走佇列原本要給常駐 worker 的 run，它錄製的輸出也永遠讀不到，因此 `record_outputs` 對它會被拒絕。

**記載於**[遠端 Worker：一次性 worker](/usage/remote-workers#one-shot-workers)與 [SLURM](/usage/remote-workers#slurm)。

### DDP 在 run 子程序內 {/* #ddp */}

**決策。** 在有兩張以上 CUDA 卡的 worker run 上，`strategy: ddp` 的 `TrainingLoop` 會在這些卡上訓練同一個模型（[#792](https://github.com/CodefyUI/CodefyUI/issues/792)）。run 子程序為每張卡啟動一個 rank；各 rank 透過 run 目錄中的 `torch.distributed.FileStore` 會合，只經由 loopback 介面通訊（`GLOO_SOCKET_IFNAME` 與 `NCCL_SOCKET_IFNAME`）。rank 0 執行整張圖，並擁有通往 agent 的 pipe；其他 rank 只執行訓練迴圈與供給它的節點。其他情況（畫布、伺服器程序、單張卡、匯出的腳本）下，訓練迴圈以單一程序訓練並發出警告。

**理由。** 對 agent、協定與 Runs 面板而言，DDP run 仍是一個 run、一份事件日誌。以檔案會合不需要監聽 socket，因此 worker 仍不開啟任何其他機器連得到的埠（不變量 4），也不必尋找並占住空閒的埠。匯出的腳本維持單一程序的路徑，匯出面不受影響（不變量 2）。

**排除的方案。** 在空閒的 loopback 埠上以 TCP 會合，也就是 #792 最初的規格：在單一機器上，它只多出一個監聽 socket 與搶埠的問題。跨機器訓練。在伺服器程序內執行 DDP，這會把多個 rank 放進提供 UI 的程序。

**記載於**[多 GPU 執行：用多張卡訓練一個模型](/usage/multi-gpu#ddp)與[執行架構：隔離](./execution-architecture#isolation)。

### 指紋與來源紀錄是兩份紀錄 {/* #fingerprint-and-provenance */}

**決策。** 環境指紋記錄執行 run 的程式碼與環境：CodefyUI 版本與 commit、Python、torch build、加速器、plugin 的 revision 與內容摘要，以及一份固定清單中的環境變數，在 run 開始時記錄。worker run 由 run 子程序回報自己的指紋，取代 agent 的指紋（`execution_facts.fingerprint_source`）。節點來源紀錄（provenance），也就是圖讀取了哪些資料集檔案與模型權重，只在 run 選項 `provenance`（`identity` 或 `content`）開啟時記錄，每個讀取節點一筆（[#344](https://github.com/CodefyUI/CodefyUI/issues/344)）。

**理由。** 指紋成本低，同一個環境上的每個 run 都相同，因此摘要相同就代表環境相同。資料識別需要讀檔，成本隨資料集變大，因此需要明確開啟；stat 識別（路徑、大小、修改時間）與 sha256 內容雜湊分開標示，兩者永遠不會被拿來比對。子程序回報自己的指紋，因為 worker 的 checkout 或 plugin 可能在 agent 啟動後改變。

**排除的方案。** 把資料識別併入指紋：環境摘要會隨每個資料集改變，每個 run 也都要計算雜湊。在送出時記錄指紋：queued run 可能在等待期間遇到伺服器更新，或在 worker 上執行，排入佇列時的環境與實際執行的環境可能不同。

**記載於** [API 參考：執行環境](./api-reference#run-environment)與[節點來源紀錄](./api-reference#node-provenance)。

### 由伺服器強制執行預算的 study {/* #studies */}

**決策。** study（多個變體 x 多次重複，以一個確切的目標排名）存放在資料庫中，由伺服器的 study coordinator 經由一般的 run 佇列驅動。它的預算（`max_runs`、`max_parallel_runs`、`max_run_seconds`、`max_active_seconds`）由伺服器強制執行，不論有沒有人在看；預算用完時 study 會暫停並保留所有結果，以提高後的預算繼續就會接著執行（[#343](https://github.com/CodefyUI/CodefyUI/issues/343)）。`options.target` 會套用到每個子 run，因此放到 worker 上的子 run 可以平行執行。

**理由。** 放在瀏覽器分頁中的實驗，分頁一關就結束。只有客戶端檢查的預算，在客戶端離開的那一刻就失效。只有每次排定的重複都成功的變體才會排名，這比 sweep 的規則嚴格，因為 study 的排名可能用來推舉變體。

**排除的方案。** 為 sweep 加上重複次數：sweep 會以失敗的 run 達到的數值排名，也不保存每次重複的 attempt，而重啟後的復原需要這些資料。

**記載於**[實驗研究（Studies）](/usage/studies)與[執行佇列：Studies](/usage/run-queue#studies)。

### Plugin job 只執行核心程式碼 {/* #plugin-jobs */}

**決策。** plugin 透過 `/api/plugins/{plugin_id}/jobs` 送出持久化的工作，範圍限於它的 id，並且必須帶 idempotency key。核心只註冊一種 kind，`study`，其 spec 就是 `POST /api/studies` 的 body；job 就是 `owner` 為該 plugin 的 study。plugin 提供的伺服器端 handler（Graph Copilot contract 中的 `AgentJobHandler`）不會被載入。

**理由。** Graph Copilot 在計畫核准後執行的是確定性的 study，伺服器上不需要 plugin 的程式碼。規劃與模型對話留在 plugin 的頁面中。

**排除的方案。** 在控制面載入 plugin 自己的常駐 handler：這需要信任確認與沙箱，目前都還沒有。之後的 kind 會掛在 registry（`register_job_kind`）上。

**記載於** [API 參考：Plugin jobs](./api-reference#plugin-jobs)與[實驗研究：來自 plugin 的 study](/usage/studies#studies-from-plugins)。

## 決策紀錄 {/* #decisions-log */}

| 日期 | 決策 |
|------|----------|
| 2026-10-15 | 把執行從控制面拆出；保留 in-process 路徑作為預設，也是唯一的單節點路徑。 |
| 2026-10-15 | 網頁優先：Workers 面板、「新增 worker」對話框與 run 目標選擇器都在瀏覽器中；CLI 只保留啟動、加入與停止。 |
| 2026-10-15 | 拓撲：任何機器都可以 `cdui start` 作為控制面；GPU 機器執行 `cdui agent start`，以瀏覽器產生的一行指令加入。連線方向一律是 worker → 控制面。 |
| 2026-10-15 | 准入維持以每個 `(worker, device)` slot 計數；依記憶體准入延後。2026-10-18 起由每張卡的**依記憶體共用**政策取代（[#788](https://github.com/CodefyUI/CodefyUI/issues/788)）。 |
| 2026-10-15 | 儲存：預設使用 NAS（mount）；也必須有不需 NAS 的 transfer 後端。 |
| 2026-10-15 | Interactive 通道留在控制面；worker 憑證透過 enrollment 取得；先支援 Linux worker，Windows 盡力而為；控制面與 worker 版本須相同；DDP 之後再做（[#792](https://github.com/CodefyUI/CodefyUI/issues/792)）。 |
| 2026-10-15 | 每個 run 一個 spawn 子程序，裝置集合以 `CUDA_VISIBLE_DEVICES` 傳入（[#771](https://github.com/CodefyUI/CodefyUI/issues/771)）；agent 程序從不 import torch。 |
| 2026-10-15 | 同機 worker 以 machine id 判定，共用的卡以 NVML UUID 判定，從不依對端位址（[#773](https://github.com/CodefyUI/CodefyUI/issues/773)）。 |
| 2026-10-15 | enrollment 完成之前，`/ws/worker` 只接受帶 session token 的 loopback 對端（[#772](https://github.com/CodefyUI/CodefyUI/issues/772)）。已由 enrollment 取代（[#780](https://github.com/CodefyUI/CodefyUI/issues/780)）：session token 仍只限 loopback，worker token 則接受任何對端。 |
| 2026-10-15 | 加入：一次性 token 換成每個 worker 獨立的憑證；session token 仍只限 loopback；身分綁定憑證，已加入的名稱只能透過重新核發 token 易手，同機需要 loopback 對端且 machine id 相符（[#780](https://github.com/CodefyUI/CodefyUI/issues/780)）。 |
| 2026-10-15 | 單一 run、sweep 與 study 共用同一條送出規則：只有啟動以來從未見過的 worker 名稱會被拒絕；目標為已知但離線之 worker 的 run 會等它回來（[#777](https://github.com/CodefyUI/CodefyUI/issues/777)）。 |
| 2026-10-18 | 多個 slot 提供的同一個實體裝置只有一份額度：slot 依 UUID 分組（同機 worker 的 CPU／`mps:0` 與伺服器的分在一起），不再對應到伺服器的 slot；取線上提供者中最小的上限、一份記憶體帳本、較嚴格的政策（[#842](https://github.com/CodefyUI/CodefyUI/issues/842)）。 |
| 2026-10-21 | DDP 是 run 子程序內的 `TrainingLoop` 策略：裝置集合有兩張以上 CUDA 卡的 worker run 每張卡一個 rank，rank 0 擁有通往 agent 的 pipe，其他 rank 只執行訓練迴圈所需的部分；其他情況（畫布、伺服器程序、匯出的腳本）訓練迴圈以單一程序訓練並發出警告。沒有多機訓練（[#792](https://github.com/CodefyUI/CodefyUI/issues/792)）。 |
| 2026-10-21 | study 由伺服器擁有，預算由伺服器強制執行；plugin 以範圍限於自身、kind 為 `study` 的 job 送出 study，不載入 plugin 的伺服器端程式碼（[#343](https://github.com/CodefyUI/CodefyUI/issues/343)）。 |
| 2026-10-22 | 環境指紋描述實際執行 run 的程序（在 worker 上是它的 run 子程序）；資料與模型的識別是另一份需明確開啟的節點來源紀錄（[#344](https://github.com/CodefyUI/CodefyUI/issues/344)）。 |

## 待決問題 {/* #open-questions */}

以目前的程式碼而言，以下問題仍待決定：

- **待決：Windows worker 支援到什麼程度。** worker 在 Linux 上開發與測試，macOS 可用，Windows 盡力而為。`cdui agent start --foreground` 可在 Windows 上執行，但把 agent 當作服務執行的方式只記載了 systemd unit（[遠端 Worker：以服務執行 agent](/usage/remote-workers#running-the-agent-as-a-service)）。
- **待決：每個 worker 多個儲存標籤。** agent 可以宣告多個標籤（`--mount` 可重複），但放置只檢查一個：控制面自己的標籤 `CODEFYUI_MOUNT_LABEL`（`data`）。同時有 NAS 與本機 SSD 的 worker，目前還無法依第二個標籤放置 run（[遠端 Worker：共用檔案系統](/usage/remote-workers#storage-shared-filesystem)）。
- **待決：讓 plugin 保持同步。** plugin 與 pack 要在每個 worker 上手動安裝，降級的 worker 會說明它缺少哪些節點型別；沒有指令能把控制面的 plugin 組合複製到 worker（[遠端 Worker：讓 worker 保持同步](/usage/remote-workers#keeping-workers-in-sync)）。

設計初期的其他問題在程式碼中已有定論：enrollment token 預設 24 小時有效、可使用一次，兩者都可在「新增 worker」對話框中逐個 token 設定；斷線 worker 的 run 會等待 `CODEFYUI_WORKER_RESUME_GRACE_S`（300 秒），重啟後的控制面會收養它們；Wave 12 的每一項（記憶體准入、worker 上的錄製輸出、一次性 worker、S3、DDP、收養、記憶體估計）都已完成。
