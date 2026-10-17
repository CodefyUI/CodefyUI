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

worker agent 會把它接到的每個 run 放在自己的子程序中執行，所以崩潰或記憶體不足被終止只會結束那一個 run，伺服器與其他 run 不受影響。本節在伺服器所在的機器上啟動它；其他地方的 GPU 伺服器請見[遠端 Worker](/usage/remote-workers#adding-a-worker)。

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
- **`any`** 依名稱順序選第一個已連線、未排空且有空 slot 的 worker，這台伺服器排最後。CUDA 裝置集合超過一張卡、或是 `cuda:0` 以外單張卡的 `any` run，只會在 worker 上執行，因為只有 worker 會為該 run 把集合從 `cuda:0` 重新編號（見下文）；伺服器見過的 worker 都不提供整個集合時，送出會以 `400` 被拒絕。

**run 內的裝置名稱。** worker 上的 run 只看得到自己的卡，並依你列出的順序從 0 重新編號。以 `--devices cuda:2,cuda:0` 為例，run 的 `cuda:0` 是機器的 `cuda:2`，run 的 `cuda:1` 是機器的 `cuda:0`。**device** 參數為 `auto` 的節點跟隨 run 的裝置（第一張卡）；要把節點放到集合中的另一張卡，把它的 **device** 設為那張卡在 run 內的名稱，此例為 `cuda:1`。run 會把對應關係記在 `execution_facts` 的 `visible_devices` 與 `device_aliases`。請參閱[裝置後端](/advanced/device-backends#addressing-one-card-out-of-several)。

**Sweep 與 study。** `POST /api/sweeps` 與 `POST /api/studies` 會把 `options.target` 傳給每個子 run。使用 `"target": {"worker": "any"}` 且不帶 `devices` 時，每個子 run 取下一個符合 sweep 裝置種類的空 slot，所以在兩個各有一張卡的 agent 上，四個 variant 的 sweep 會同時跑兩個；worker 上帶種子的子 run 各在自己的程序中，可以並行。單一 run 會被拒絕的 target，會讓整個 sweep 或 study 在排入任何東西之前以 `400` 被拒絕。每個 variant 與 attempt 都會回報 `worker_id`、`devices` 與 `queue_key`。**新增掃描**對話框有與 **Queue on...** 相同的目標與裝置選擇器：本伺服器、已連線的 worker 加上裝置集合，或任一可用加上裝置種類。維持本伺服器與分頁的裝置時，sweep 會像以往一樣送出，不帶 target。sweep 檢視會顯示每個 variant 的 Worker 與裝置。study 請透過 API 設定 `target`。請參閱[執行佇列：Sweep](./run-queue#sweeps)。

## Slot 與上限 {/* #slots-and-limits */}

**slot** 是一台機器上的一個裝置。加速器的每個 slot 一次容納一個 run，CPU 容納兩個，其餘依送出順序等待。裝置集合 run 需要集合中每個裝置都有空 slot，並一次全部占用；它在佇列最前面等待時會保留這些 slot，所以一連串單卡 run 無法讓它餓死。

`CODEFYUI_RUN_QUEUE_MAX_CONCURRENT` 修改單一 slot 的上限。只寫裝置指的是這台伺服器；`worker/device=N` 指的是某個 worker：

```bash
CODEFYUI_RUN_QUEUE_MAX_CONCURRENT="cuda:0=2,gpu-box/cuda:1=3" cdui start
```

同一台機器上的 worker，對實體上相同的每張卡（以卡的 UUID 比對）共用這台伺服器的 slot，因此在上限為 1 時，伺服器上的 run 與該 worker 上的 run 不會同時拿到同一張卡。這種卡套用伺服器的上限，針對它的 `worker/device` 覆寫會被忽略並發出警告。提高上限會讓多個 run 共用一張卡的記憶體，但沒有任何機制檢查它們是否放得下。若要依記憶體共用 worker 的卡，請見[共用一張卡](#sharing-a-card)。細節見[執行佇列：設定](./run-queue#configuration)。

## 共用一張卡 {/* #sharing-a-card */}

worker 的卡預設一次只接收一個 run（**獨佔**）。要讓多個 run 共用，在 [Workers 面板](#the-workers-panel)把該卡設為 **依記憶體共用**。這個設定依 worker 與卡儲存，重新連線與伺服器重啟後都會保留。只有 CUDA 與 MPS 卡可以共用，而且只能在 worker 上：這台伺服器上的 run 與這裡所有其他 run 共用一個程序，沒有辦法把它限制在預算內。

送到共用卡的 run 要宣告**記憶體預算**，單位是 GiB、針對其裝置集合中的每張卡：`target.memory_gib`，或[排入佇列...](#queue-on)中的 **記憶體預算** 欄位。若同一張圖先前在相同張數的卡上有成功結束的 run，欄位會預先填入它的 CUDA 峰值再加 20%（兩張卡的 run 的每卡峰值，不代表同一張圖在一張卡上的用量；MPS 沒有峰值，只有結束時驅動程式持有的量，所以 MPS 的 run 要自行宣告預算）。「同一張圖」指的是執行時的圖：節點類型、影響記憶體用量的參數，以及連線。節點 id 與位置、標籤、seed，以及執行多久（`epochs`、`max_steps`、`lr`）都不算在內，因此換個 seed 或增加 epochs 重跑時，仍會找到先前的峰值。

沒有這樣的 run 時，欄位會填入**靜態估計值**，在對話框中標為**估計值**，不會當成峰值呈現：圖中模型的參數量乘上訓練時每個參數占用的量（權重、梯度與優化器狀態：Adam 為 16 bytes，帶 momentum 的 SGD 為 12 bytes，不訓練的圖為 4 bytes），加上依圖中 batch size 一個訓練步驟的激活，或以評估節點自己的 batch 跑一次評估的激活（取較大者），再加 128 MiB 給 CUDA 函式庫的工作區，最後再加 20% 給配置器的取整。伺服器在 PyTorch 的 `meta` 裝置上建立每個模型來計算參數量（只有形狀、不占記憶體，在獨立的執行緒中執行並有時間上限），並用一個 `meta` 輸入跑一次前向傳遞來估計激活，輸入形狀取自模型的第一層與圖中的資料節點（`image_size`、`seq_len`、已知的資料集）。在內建的訓練範例上，峰值超過 1 GiB 的 run，估計值為實測峰值的 1.2 到 1.8 倍；小型範例的峰值大多是固定開銷，估計值最多多出 0.25 GiB。必須略過部分圖的估計（無法計算的模型節點，例如外掛提供的節點；推不出模型激活的輸入形狀；或圖中沒有 batch size）會以**粗估**顯示並附上原因，且不會用作預設值。

未修改的建議值不會送出：伺服器會套用同一個預設值，Runs 面板的預算欄也會標出它來自哪個 run，估計值則標為「（估）」。在預設預算下記憶體不足的 run 會說明這一點，並請你宣告更大的 `target.memory_gib`。既沒有峰值也沒有可用的估計值時，伺服器會拒絕該 run 並要求填寫預算。sweep 或 study 會在寫入任何 run 之前整批檢查，每個 run 依它自己的 `target.wait`：只要其中一個 run 會被拒（沒有預算、預算比它能用的每張卡都大，或在 `wait: false` 時目前放不下），整個 sweep 或 study 就會被拒。

```json
{"options": {"target": {"worker": "gpu-box", "devices": ["cuda:0"], "memory_gib": 6}}}
```

伺服器為每張卡維護一份帳本：worker 心跳回報的總記憶體與可用記憶體（可用量計入卡上所有程序，包括其他使用者的工作）、已指派但尚未開始的 run 的預算（**已保留**），以及已開始的 run 的預算（**執行中**）。CUDA 卡上的 run 還會占用自己的 CUDA context（cuBLAS handle、已載入的 kernel），下面的上限管不到這部分，所以伺服器把每個 run 算成預算加上 `CODEFYUI_EXECUTION_CUDA_CONTEXT_MIB`（預設 768 MiB；在一張 CUDA 12 的卡上實測約 650 MiB）。這個總和不超過 `min(free - reserved, total - reserved - running)` 時就放得下，拒絕訊息也會兩者都列出：`needs 6.0 GiB (+0.8 GiB CUDA context), ...`。worker 會在 run 開始前把它的配置器限制在預算內（`torch.cuda.set_per_process_memory_fraction`），所以超出預算的 run 只會讓自己記憶體不足，同一張卡上的其他 run 繼續執行。

預算放不下時：

- 指定 worker 的 run 會立即被拒並說明原因：`needs 12.0 GiB, cuda:1 on gpu-box has 6.3 GiB free (runs a1b2c3d4, c3d4e5f6)`。加上 `target.wait: true`（對話框中的 **目前放不下時在佇列中等待**）則改為等待；
- `any` 的 run 會等到有卡放得下，除非它的預算比每張候選卡都大（拒絕），或 `target.wait` 為 `false`（拒絕而不等待）。送往 one-shot pool（`target.pool`）的 run 以同樣方式判斷，候選卡是 one-shot worker 的卡；
- 等待記憶體的 run，Runs 面板會在它的 worker 下方說明。

沒有預算的 run 不會被送到共用卡；共用卡上沒有預算的 run（同機 worker 卡上的畫布 run，或在該卡改為共用之前就開始的 run）會獨佔該卡直到結束。Runs 面板的 **預算** 欄顯示每個 run 的預算，Workers 面板則畫出每張共用卡的記憶體：使用中、已保留，以及新預算可用的量（已扣除 CUDA context）。在卡為獨佔時排入等待的 run，會在該卡改為共用時取得其圖的預設預算，或之後在同一張圖於相同張數的卡上成功執行時取得；在那之前它會繼續等待，Runs 面板會說明（想早點開始，可以取消後帶著預算重新排入）。依預算准入的 run 不會在無法限制它的 agent 上啟動：若 worker 在這段期間以舊版 agent 重新連線，該 run 會失敗並說明原因。伺服器重啟後接手的 run（[遠端 Worker](/usage/remote-workers)）一開始就會在它的卡上重新以執行中計入其預算。

## Workers 面板 {/* #the-workers-panel */}

從 Runs 面板標題列的 **Workers**，或 **Settings → 執行 → Workers → 管理**開啟。開啟期間每 5 秒更新一次，每個 worker 會顯示：

- 狀態（**Online**、心跳延遲或節點型別與伺服器不同時為 **Degraded**、**Draining**、**Disconnected**）、版本以及是否與伺服器相同、環境摘要、節點型別是否與伺服器相同（以及哪些不同）、上次連線被拒的原因、mount 標籤（見[遠端 Worker：儲存](/usage/remote-workers#storage)），以及上次心跳距今多久；
- 每個裝置在最近一次心跳時的可用與總記憶體、slot 的已占用 / 上限，以及 CUDA 或 MPS 卡接收 run 的方式（**獨佔** 或 **依記憶體共用**，見[共用一張卡](#sharing-a-card)），共用卡另有記憶體長條；
- 它正在執行的 run，每個都連到 Runs 面板中的該 run。

伺服器拒絕過（例如版本不同）且沒有卡片的 agent，會列在 **被拒的連線** 下並附上原因（[遠端 Worker，讓 worker 保持同步](/usage/remote-workers#keeping-workers-in-sync)）。

**排空**讓新的 run 不再被放到該 worker，它正在執行的 run 則會跑完；**取消排空**恢復原狀。worker 正在排空或離線時，queued run 會等待，Runs 面板會在該 run 的 worker 下方說明原因。**新增 worker** 會產生另一台機器用的加入指令，或顯示上面那段同機指令與其參數。**重新核發** 讓 worker 換新憑證，**移除** 會撤銷憑證、中斷連線，並讓佇列中等它的 run 失敗（[遠端 Worker](/usage/remote-workers)）。

## 畫布的行為 {/* #the-canvas */}

畫布上的 **Run** 一律在伺服器程序內、以圖的裝置（否則為 Settings 中的裝置）執行，從不送到 worker。它跳過佇列，所以即使某張卡的上限是 1，它仍可能與 queued run 共用那張卡。它會在多次點擊之間保留分頁的快取輸出與模型權重，並為 Inspector 記錄輸出，這些 worker run 都做不到。要把分頁中的圖交給 worker，請使用[排入佇列...](#queue-on)。

## Runs 面板中的峰值記憶體 {/* #peak-memory */}

run 結束後，詳細資訊標題會顯示 CUDA 的**峰值記憶體**（配置器的最高水位，滑鼠停留可看到每張卡），或 MPS 的**結束時佔用記憶體**（驅動程式仍持有的量，含快取；MPS 沒有峰值計數器）。worker 上的 run 在自己的子程序中單獨執行，因此數字是精確的。在伺服器程序內執行的 run 共用一個 CUDA context，若同時有另一個 run 使用同一張卡，數字只是近似值。提高某張卡的上限之前，可以用已結束 run 的峰值判斷兩個這樣的 run 是否放得進一張卡。

## 限制 {/* #limits */}

- **單一 run 不會跨卡訓練。** 裝置集合讓一個 run 看得到多張卡，節點也可以用 run 內名稱固定到這些卡上，但目前還沒有 `DistributedDataParallel` 策略（[#792](https://github.com/CodefyUI/CodefyUI/issues/792)）。
- **只有 worker 有依記憶體的准入。** 這台伺服器上的卡以每個 slot 的數量准入：不會因為卡上記憶體不足而拒絕 run，塞太滿的卡會以記憶體不足錯誤失敗。worker 的卡可以[依記憶體共用](#sharing-a-card)。請參閱[訓練記憶體](/advanced/training-memory)。
- **帳本計算的是預算，不是 run 實際用量。** 執行中的 run 以它的預算計算，可用記憶體則來自 worker 的心跳（每 10 秒一次），針對整張卡：其他使用者的工作在兩次心跳之間占用的記憶體，要到下一次心跳才看得到，也不會量測每個 run 的用量。
- **機器之間是明文 HTTP。** 遠端 worker 以控制面提供的協定加入與連線；沒有 TLS proxy 或 SSH 通道時，它的 token 與 run 的輸入會以明文經過網路（[遠端 Worker：信任](/usage/remote-workers#trust-between-agent-and-control-plane)）。
- **worker 上不擷取輸出**（`record_outputs`，[#789](https://github.com/CodefyUI/CodefyUI/issues/789)），也不保留權重：每個 worker run 都從頭開始。
- **長時間斷線會結束 run。** 失去連線的 worker 會在 `CODEFYUI_WORKER_RESUME_GRACE_S`（5 分鐘）內繼續它的 run；超過後它們以 `interrupted` 結束（[遠端 Worker：連線中斷時](/usage/remote-workers#when-the-connection-drops)）。

## 另請參閱 {/* #see-also */}

- [執行佇列](./run-queue)——佇列順序、`cdui run`、`target` 選項與 worker 的完整說明
- [執行架構](/advanced/execution-architecture)——控制面、executor、worker 協定與不變量
- [裝置後端](/advanced/device-backends)與[訓練記憶體](/advanced/training-memory)——裝置、指定單張卡與記憶體不足錯誤
