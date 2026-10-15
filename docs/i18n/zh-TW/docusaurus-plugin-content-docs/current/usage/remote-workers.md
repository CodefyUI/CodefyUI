---
sidebar_position: 3.7
title: 遠端 Worker
description: 在另一台機器上執行圖：用 Workers 面板產生的一次性 token，把 GPU 伺服器加入為 worker agent，或讓伺服器透過 SSH 啟動它。
---

# 遠端 Worker

Worker agent（`cdui agent start`）替 CodefyUI 伺服器（控制面）執行圖，每個 run 在自己的子程序裡跑。它可以與伺服器在同一台機器上（[CLI 指令，Agent 指令](/getting-started/cli-commands#agent-commands)），也可以在另一台機器上，也就是本頁的內容。Agent 一律主動連到伺服器，worker 機器不開任何連接埠。Run 依它的 target 交給 worker（[Run 佇列，Worker](/usage/run-queue#workers)）。

## 加入 worker {/* #adding-a-worker */}

1. **讓伺服器連得到。** `cdui start` 預設只監聽 `127.0.0.1`，其他機器無法連線。請以 `cdui start --host 0.0.0.0` 啟動。伺服器會接受自己各網路介面的位址（例如 `192.168.1.5:8000`）作為 `Host`；若 worker 改用主機名稱連線（`gpu-gw.lan:8000`、VPN 名稱），請把它加入 `CODEFYUI_EXTRA_ALLOWED_HOSTS`。任何連得到該連接埠的人都能操作這台伺服器：請只在可信任的網路中使用，或放在 TLS proxy 後面（[部署](/usage/deployment)）。
2. **產生 token。** 開啟 **Workers**（設定的「執行」，或 Runs 面板標題列），按 **新增 worker**，停在 **另一台機器**。位址欄預設為此頁面使用的位址；若是 `127.0.0.1` 或 `localhost`，請改選伺服器的其他位址，或輸入 worker 連得到的位址。選擇 token 的有效期限（預設 24 小時）與可供幾個 agent 使用（預設 1），再按 **產生 token**。
3. **在 worker 上執行指令**，使用同版本的 CodefyUI 安裝：

   ```bash
   cdui agent start --control http://192.168.1.5:8000 --token abcdef.0123456789abcdef
   ```

   Agent 會用 enrollment token 換取自己的 worker 憑證，存在 `<user-data>/agent/<host>-<port>.token`（只有該作業系統使用者可讀）。幾秒內這個 worker 就會出現在 Workers 面板。`--name` 指定 worker 名稱（預設為主機名稱）。

Enrollment token 只顯示一次，伺服器只保存它的雜湊值。已用完、已過期或已撤銷的 token 會被拒絕，訊息會說明是哪一種。已發出的 token 會列在對話框中，直到你刪除為止。

**重新啟動。** `cdui agent start --control <url>` 會找到已存的憑證並用它重新連線，不需要 token。伺服器重啟後也記得已加入的 worker：在它連線前列為 `offline`，指定它的 run 會在佇列中等待。`cdui agent forget <url>` 刪除已存的憑證，下次啟動時重新加入。

**移除 worker。** Workers 面板的 **移除** 會撤銷該 worker 的憑證、中斷它的連線（執行中的 run 以 `interrupted` 結束，佇列中的 run 失敗）並從清單移除；之後它的 agent 都會被拒絕。

**重新核發憑證。** 名稱一經加入即被占用：一般 token 無法再次加入該名稱（伺服器回 `409`）。要讓某個 worker 換新憑證（重灌機器、遺失憑證檔），請按它卡片上的 **重新核發**：在那裡產生的 token 只能加入該 worker，指令也會帶上它的 `--name`。使用後舊憑證即失效。若該機器上還留有舊憑證檔，請先在那裡執行 `cdui agent forget <url>`。

沒有 TLS 時，token 以及送到 worker 的每個 run 的輸入都會以明文經過網路；若發生在其他機器的連線上，伺服器會記錄警告。[Agent 與控制面之間的信任](#trust-between-agent-and-control-plane) 說明兩端各自檢查什麼。

## 儲存：沒有共用檔案系統 {/* #storage-no-shared-filesystem */}

會讀取上傳檔或寫出檔案的 run，也可以交給與這台伺服器沒有任何共用資料夾的 worker。此時伺服器就是檔案庫：把 run 用到的上傳檔送給 worker，並接收 run 寫出的檔案。有共用資料夾的 worker（[儲存：共用檔案系統](./run-queue#storage-shared-filesystem)）仍然優先；只有 run 指定的 worker 沒有共用資料夾，或 `any` 找不到有空的此類 worker 時，才會走這條路。

**輸入。** 能用這種方式送過去的，是在檔案欄位中選取的上傳檔：CSVReader 等資料檔欄位、ImageReader、ModelLoader、CheckpointLoader。指派時，伺服器以 SHA-256 雜湊值標示每個檔案。Agent 把下載過的檔案存在 `<user-data>/agent-cache/<sha256>`，之後使用同一個檔案的 run 不會再下載；快取中沒有的檔案，以 worker 的 token 從伺服器既有的下載路徑取得，並核對雜湊值。下載期間，執行任務面板的紀錄會顯示 `fetching inputs 2/3: train.csv (1.2 MiB, fetched)`（kind 為 `transfer_progress` 的 `run_warning`）；顯示 `cached` 而非 `fetched` 表示取自快取。下載期間按停止，run 以 `cancelled` 結束。

**Run 的工作資料夾。** 每個 run 在 worker 上有自己的資料夾 `<user-data>/agent-runs/<run id>/data`，結構與 `backend/data` 相同（`files/`、`images/`、`models/`、`media/`），每個輸入檔以圖中使用的名稱複製一份放在裡面（節點覆寫輸入檔也碰不到快取）。Worker 自己的資料夾不會被動到，只有 `cache/` 資料夾（LMTokenizedDataset 的 token 區塊）由各 run 共用。Dataset、HuggingFaceDataset 下載的資料集仍放在 worker 自己的快取，與在伺服器上相同。

**輸出。** 在 run 的結束訊息送到伺服器之前，agent 會上傳：

- run 記錄為 artifact 的每個檔案（定期 checkpoint、TensorBoard 資料夾則逐檔上傳），在 run 回報時上傳到 `<資料夾>/runs/<run id>/<kind>/<name>`；
- run 結束後，它在 `models/`、`media/`、`output/` 底下寫出的其他檔案（CheckpointSaver、ModelSaver、VideoWrite、ImageWriter），以及被它覆寫的輸入檔，上傳到這台伺服器自己的資料夾、相同的相對路徑，也就是同一個節點在本機 run 時寫入的位置（`CODEFYUI_MODELS_DIR` 中的 `models/best.pt`）。該處已有不同的檔案時會保留原檔，新檔加上後綴（`best-1.pt`）；內容相同則不重複存放。這些檔案的 kind 為 `output`；
- 最後是被停止的訓練迴圈寫出的 checkpoint，存到 run 的資料夾，讓已取消 run 的續訓點留在伺服器上。

上傳與 run 並行，網路慢不會拖住 run；上傳完成後才回報 run 結束。`GET /api/runs/{id}/artifacts` 以 `backend: "transfer"` 和這台伺服器上的路徑列出，執行任務面板在每一筆旁提供 **下載**（`GET /api/runs/{id}/artifacts/{artifact_id}/download`）。超過這台伺服器上傳上限（`CODEFYUI_MAX_UPLOAD_SIZE`，500 MB）的檔案會留在 worker 上，以它在 worker 上的路徑列出，並附上 `transfer_failed` 警告。在最後一批上傳期間按停止，尚未送出的檔案就不再上傳，並以一則 `transfer_failed` 警告說明有幾個留在 worker 上。

**清理。** run 自己的資料夾屬於該 run：run 保留機制（`CODEFYUI_RUN_RETENTION_KEEP_LAST`）或 **刪除** 移除 run 時，它的 `runs/<run id>` 資料夾也一起刪除。想保留的檔案請下載或另行複製。以 `interrupted` 結束、之後被保留機制移除的 run 會保留 `checkpoint` 資料夾，與本機 run 保留續訓點的做法相同。放進資料夾的檔案屬於你，與本機 run 相同：兩者都不會刪除它們。在 worker 上，run 的資料夾跟著 agent 自己的 run 保留機制；快取不會自動清理，刪除 `agent-cache` 是安全的（需要時會重新下載）。

**限制。**

| 設定（在伺服器上） | 預設值 | 說明 |
| --- | --- | --- |
| `CODEFYUI_TRANSFER_ENABLED` | `1` | 設為 `0` 即關閉：使用檔案的 run 必須交給有共用資料夾的 worker，與先前相同 |
| `CODEFYUI_TRANSFER_MAX_INPUT_BYTES` | 2 GiB | 一個 run 的輸入檔總大小上限 |
| `CODEFYUI_MAX_UPLOAD_SIZE` | 500 MB | worker 可上傳的單一輸出檔上限 |

無法以此方式處理的 run，若指定了沒有共用資料夾的 worker，會在送出時被拒絕，訊息會說明原因：讀取節點用的是一般路徑而非上傳檔（FileReader、ImageFolderDataset、ImageBatchReader、VideoLoad、讀取資料夾的 DocumentLoader）、這台伺服器上沒有該上傳檔，或輸入超過上限。例如：``worker gpu-box-1 has no `data` mount; add `--mount data=...` to its agent, or make the run fit the transfer backend: its inputs add up to 3.10 GiB, over the transfer limit of 2.00 GiB (CODEFYUI_TRANSFER_MAX_INPUT_BYTES)``。

輸入與輸出送往 agent 連線的同一個位址，帶著它的 worker token，並做相同的憑證檢查（釘選的憑證對它們同樣生效）：沒有 TLS 時，它們會以明文經過網路。

## 連線中斷時 {/* #when-the-connection-drops */}

連線中斷的 worker 會繼續執行它的 run，並保留事件直到這台伺服器收到為止。在這台伺服器上，這些 run 顯示為 `disconnected`，並在 `CODEFYUI_WORKER_RESUME_GRACE_S`（5 分鐘）內保留名額；在此期間重新連線，run 會接續執行，紀錄沒有缺漏，超過時間則以 `interrupted` 結束。各種結果，以及筆電休眠的影響，見[連線中斷時](./run-queue#when-the-connection-drops)。
## Agent 與控制面之間的信任 {/* #trust-between-agent-and-control-plane */}

送到 worker 的 run 會帶著填回 secret 參數值（API 金鑰、密碼）的圖，所以兩端都必須確認對方是誰。控制面以 worker 憑證認出 agent；agent 則以 TLS 憑證認出控制面，加入請求與每一條 `/ws/worker` 連線都做同一個檢查。

CodefyUI 本身不提供 TLS；請把它放在 TLS proxy 後面（[部署](/usage/deployment)），並給 worker `https://` 位址。

**Worker 的 CA 信任的憑證**（公開 CA，或安裝在 worker 上的公司 CA）：不需要做任何事。Agent 每次連線都像瀏覽器一樣檢查憑證機構與主機名稱，且不記錄任何東西，所以憑證續期（即使換了金鑰）仍可繼續使用。

**自簽憑證，或 worker 沒有的 CA：釘選它的金鑰。** `--ca-fingerprint` 接受伺服器憑證公鑰（SubjectPublicKeyInfo）的 SHA-256，寫成 `sha256:<64 個十六進位字元>`，做法與 `kubeadm join --discovery-token-ca-cert-hash` 相同：

```bash
cdui agent start --control https://cp.lab --token abcdef.0123456789abcdef \
  --ca-fingerprint sha256:96d448128bb7dfbceea1a628393cfdd0f1f86716ad81deb4fab0c0fe281a0690
```

參數名稱沿用，但它釘選的是伺服器自己（leaf）憑證的金鑰，從不是 CA 的：在任何 Python 版本上，釘選值只比對 TLS 交握證明過其金鑰的那張憑證。有釘選時，agent 不檢查憑證機構，也不檢查主機名稱。不符的憑證會在送出 token 之前就被拒絕，並印出兩個指紋，agent 以代碼 `2` 結束。釘選值會記在憑證檔（`<user-data>/agent/<host>-<port>.token`），之後啟動不帶參數也會沿用；存下的值若不是合法指紋，會在日誌中指名該檔案回報並忽略（worker 憑證仍可使用）。

要取得指紋，請在 proxy 主機上對 proxy 使用的憑證檔執行（`fullchain.pem` 中第一張是 leaf）：

```bash
openssl x509 -in /path/to/fullchain.pem -pubkey -noout | openssl pkey -pubin -outform der | openssl dgst -sha256 | awk '{print "sha256:" $NF}'
```

位址是 `https://` 時，Add worker 對話框會顯示這個指令。若伺服器自己讀得到憑證檔，把 `CODEFYUI_TLS_CERT_FILE` 設為它的路徑：對話框就會顯示指紋，產生的指令也會帶上 `--ca-fingerprint`（伺服器只在檔案改變時重新讀取）。Worker 的 CA 信任該憑證時請不要設定，以免指令釘選一把續期時就會換掉的金鑰。

**Worker 的 CA 不接受、又沒有釘選的憑證**不會被使用：加入會失敗；執行中的 agent 會記錄錯誤與原因（第一次會附上伺服器出示的指紋，供你與對話框比對後用 `--ca-fingerprint` 帶入），並以平常的退避間隔重試，所以修好 proxy 或更新 CA 清單後不必重啟。

**已釘選的憑證換了金鑰之後**，先在 proxy 主機上確認新指紋，然後擇一：

- 執行 `cdui agent start --control <url> --ca-fingerprint <新指紋>`：保留 worker 憑證，連線成功後記錄新的釘選值；或
- 執行 `cdui agent forget <url>`：刪除憑證與其中的釘選值，再用新的 token 重新加入。

續期的憑證只有在沿用金鑰時（certbot 的 `--reuse-key`）指紋才不變。

**未加密的 HTTP。** 以 `http://` 位址連到另一台機器時，`cdui agent start` 會印出一次警告：worker token 與每個指派中的 secret 參數都以明文傳輸。Agent 仍會連線。Workers 面板會把這種 worker 標成 **未加密 HTTP**：它的連線從另一台機器未加密地抵達伺服器。伺服器依連線的 scheme（只有 uvicorn 回報 `wss` 才算，uvicorn 只對 `--forwarded-allow-ips` 列出的 proxy 這麼做；CodefyUI 本身從不讀 `X-Forwarded-Proto`）與對端位址判斷：loopback 對端，例如同一台機器的 agent 或 SSH 通道的本機端，不會被標記，因為跨越網路的不是那條 socket。同樣的理由，`http://127.0.0.1` 位址不會讓 agent 印出警告。

不涵蓋：mutual TLS（agent 由 worker 憑證識別）、由 CodefyUI 管理的憑證機構，以及使用者身分。

## 讓 worker 保持同步 {/* #keeping-workers-in-sync */}

worker 必須執行與伺服器相同的 CodefyUI 版本，也應該認得相同的節點。agent 每次連線時，伺服器都會檢查這兩項。

**版本。** 版本不同的 agent 會被拒絕，原因中列出雙方版本，例如 `version mismatch: worker 3.1.0, control plane 3.1.2`。agent 會停止而不重試，Workers 面板則在 **被拒的連線** 下列出該名稱與原因，所以一直沒出現的 worker 會說明原因；試圖搶用已連線 worker 名稱的 agent 會連同位址列在那裡，不會顯示在該 worker 的卡片上。版本會在 agent 第一則訊息中最先被讀取，所以即使訊息格式已改變的 agent 也會得到這個原因。在 worker 上安裝相同版本後重新啟動 agent；該名稱連線成功後，拒絕紀錄就會清除。面板會保留每個名稱最近一次的拒絕，直到伺服器重新啟動。

**節點型別。** 外掛、Package Center 套件與自訂節點都是每台機器各自安裝。agent 啟動時，一個短暫的輔助程序會以 run 的方式載入所有節點，回報節點型別與其定義的雜湊（節點目錄）。雜湊與伺服器不同時，worker 會標為 **Degraded**，卡片列出 worker 缺少的型別，以及只有它有的型別。降級的 worker：

- 缺少伺服器有的型別時，永遠不會收到 target 為 `any` 的 run（只比伺服器多出型別的 worker 仍可接收）；
- 只有在圖用到的每個節點型別它都有時，才會收到 run：包括展開 preset 與 subgraph 之後的節點，以及 **Map** 節點所執行的每個已安裝 preset 中的節點。否則送出時就會被拒絕並列出缺少的型別；佇列中的 run 在送出前也會以同樣方式失敗，以防 worker 在等待期間以較少的型別重新連線。

要解除降級，請在 worker 上安裝相同的外掛與套件（`cdui plugin install ...`、Package Center、相同的自訂節點檔）並重新啟動 agent：它只在啟動時回報節點。輔助程序失敗的 worker 不會回報節點目錄：它的卡片會顯示警告，`any` 會略過它，指定它的 run 則不經檢查直接送出（agent 的 log 會說明輔助程序失敗的原因）。把伺服器的外掛組合複製到 worker 的指令（`cdui agent sync`）尚在規劃中。

**環境指紋。** 同一個輔助程序也會收集 worker 的[環境指紋](/advanced/api-reference#run-environment)（CodefyUI commit、Python、torch build、加速器、外掛版本、影響數值的環境變數）。Workers 面板顯示它的摘要，worker 執行的每個 run 都會記錄這個摘要（重新連線時未回報指紋的 worker 會顯示「未回報」，不會沿用舊的摘要），所以 Runs 面板會指出實際執行它的環境。指紋描述的是 agent 的環境，也就是它的 run 子程序所繼承的環境，在 run 的裝置集合縮小 `CUDA_VISIBLE_DEVICES` 之前。

## 透過 SSH 連線 {/* #connecting-over-ssh */}

若 GPU 伺服器只能用 SSH 連到，例如筆電在 NAT 後面、伺服器只開放 22 埠，可以讓伺服器自己啟動 worker。它會用 `ssh` 連到 GPU 伺服器、在那裡執行 `cdui agent start`，agent 再經由同一個 SSH 連線中的反向通道（`ssh -R`）連回來。兩台機器都不需要開連接埠，伺服器也可以維持只監聽 `127.0.0.1`。

**開始之前：**

- GPU 伺服器上已用 `install.sh` 安裝相同版本的 CodefyUI。伺服器不會替你安裝。
- 伺服器所在的機器 PATH 上有 OpenSSH 用戶端（`ssh`）。macOS 與 Linux 內建；Windows 請加入選用功能 **OpenSSH 用戶端**，並在 PATH 含有它的工作階段中啟動伺服器。
- 在伺服器機器上、以執行 `cdui start` 的作業系統使用者開終端機執行 `ssh <target>`，**不會出現任何提示**就能登入。ssh 以 `BatchMode=yes` 執行：不能用密碼，有 passphrase 的金鑰須透過 `ssh-agent`。你的 `~/.ssh/config` 會生效，所以 `Host` 別名、`ProxyJump` 與其中指定的金鑰都能用。

**新增。** 開啟 **Workers**，按 **新增 worker**，選 **透過 SSH**：

| 欄位 | 說明 |
|---|---|
| SSH 目標 | `user@host`，或伺服器帳號 `~/.ssh/config` 中的 `Host` 別名。 |
| 連接埠 | ssh 的 `-p`。保持 22 時不會傳給 ssh，所以 ssh 設定中該主機的連接埠會生效。 |
| Worker 名稱 | 選填；預設為目標的主機名稱（`alice@gpu1.lab` 會成為 `gpu1`）。已被其他 worker 使用的名稱會被拒絕。 |
| 金鑰檔 | 選填；伺服器機器上的私鑰（ssh 的 `-i`）。留空則使用你的 ssh 設定與 agent。 |
| 遠端主機上的 cdui | 遠端啟動器的路徑；留空代表 `~/CodefyUI/cdui`，也就是 `install.sh` 放置的位置。 |

按 **新增並連線**。Worker 卡片上會多一行 **SSH 通道**：ssh 執行中但 agent 尚未連線時為 **連線中**，連上後為 **已連線**，等待重試時為 **失敗** 並附上 ssh 或 agent 最後輸出的幾行（金鑰被拒、未知主機、找不到 `cdui`），停用時為 **已停止**。**最近的 ssh 輸出** 可看到更多輸出。

**實際執行的內容。** 每個啟用中的 SSH worker，伺服器都會維持一個像這樣的程序：

```bash
ssh -T -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
    -o StrictHostKeyChecking=accept-new \
    -R 127.0.0.1:18000:127.0.0.1:8000 -- alice@gpu-box \
    "sh -c '... ~/CodefyUI/cdui agent start --foreground --control http://127.0.0.1:18000 --name gpu-box'"
```

遠端指令會先把 `~/.local/bin` 加到 PATH：非互動式的 ssh shell 不會讀取你的 shell 設定檔。遠端連接埠取自 `CODEFYUI_SSH_REMOTE_PORT_RANGE`（`18000-18100`）；若該埠在遠端主機上已被占用，會改試下一個並記住。ssh 結束後，伺服器會在 2 秒後重新啟動它，間隔逐次加倍到最多一分鐘；伺服器自己重啟後也會重新連線。若遠端 sshd 根本不允許轉送（`AllowTcpForwarding no`、`PermitListen` 不含該埠；明確拒絕，或連續三個埠都失敗），通道會停在 **失敗** 並顯示原因，不再自動重試；請伺服器管理者允許轉送到 `127.0.0.1` 的 `-R` 後按 **重試**。通道列也會顯示遠端 agent 的 PID：在遠端主機上 agent 的命令列是 `python .../scripts/dev.py agent start --foreground ...`，可用 `pgrep -af 'agent start'` 找到它。

**憑證。** 每次啟動都會經由 ssh 的標準輸入送出一次性的 enrollment token（有效 10 分鐘），不會出現在任何命令列上，所以兩台機器的 `ps` 都看不到它。第一次啟動時 worker 會加入；agent 會把 worker 憑證以 `http://127.0.0.1:<遠端連接埠>` 這個位址存在遠端主機上，之後每次啟動都用它重新連線，token 則不會被使用。Worker 加入後，token 會綁定到它，只能用來重新核發這個 worker 的憑證。這個 worker 第一次使用某個埠時，會先對該位址執行 `cdui agent forget`，所以其他 worker 或其他控制面留下的憑證不會被送出。若仍送出了這種憑證（這台伺服器在該通道上拒絕它，或 agent 回報憑證屬於另一個 worker 名稱），伺服器會先執行 `cdui agent forget` 再重新啟動 agent，讓它重新加入。所有通道都從這台機器自己的位址連入，因此經由通道的加入嘗試會依通道（其遠端連接埠）分開限速，彼此之間以及與這台機器上的 agent 互不影響。

**連線中斷。** 若連線不是因停用而結束（筆電睡眠、網路中斷），遠端 agent 與它的 run 會繼續執行；這些 run 在這台伺服器上以 `disconnected` 等待並保留事件，如同[連線中斷時](#when-the-connection-drops)所述。伺服器會以相同的遠端連接埠重新啟動 ssh，遠端腳本發現 agent 仍在執行，就會接上它而不另啟一個（通道列會顯示重新接上的 PID）。Agent 會自行經由新的通道重新連線，它的 run 也會繼續。若遠端 sshd 還沒發現舊連線已消失、仍占用該埠，伺服器會先另開一個短暫的 ssh 結束那個殘留的連線。若在 `CODEFYUI_WORKER_RESUME_GRACE_S`（5 分鐘）內沒有新的連線接上，agent 會停止。遠端腳本把 agent 的 PID 與日誌存在遠端主機的 `~/.codefyui-ssh/<遠端連接埠>/`。

**停用、啟用、刪除。** **停用** 會先讓 worker 停止接新 run，再結束它的 ssh 連線；之後即使有 agent 以它的名稱連線，停用中的 worker 仍維持不接新 run。停用會經由連線通知遠端 agent 停止（沒有連線時則另開一個短暫的 ssh），所以仍在執行的 run 會以 `interrupted` 結束；若有需要，請先排空並等它完成。**啟用** 會重新啟動連線。對尚未連線的 worker，**刪除** 會結束連線並移除該項目；對已連線的 worker，**移除** 會撤銷它的憑證，並一併刪除它的 SSH 項目。

**安全性。**

- 通道只監聽遠端主機的 loopback（`-R 127.0.0.1:...`）。該主機上的每個使用者仍可連到那個埠，因此經由它的請求只能到達 `/ws/worker` 與 `POST /api/workers/enroll`；其他路徑一律回 `421`，而這兩個端點都需要 token。
- `StrictHostKeyChecking=accept-new` 會在第一次見到主機時信任它，金鑰改變時拒絕連線。設定 `CODEFYUI_SSH_STRICT_HOST_KEY_CHECKING=yes` 則要求主機已在伺服器帳號的 `known_hosts` 中。
- 伺服器使用執行它的作業系統使用者的 SSH 存取權，就像 Source Control 分頁使用 git 一樣。能操作這台伺服器的人，就能在該使用者連得到的每台主機上啟動 SSH worker。
- 通道後面的 agent 從 `127.0.0.1` 連線，但只有在它的 machine id 與伺服器相同時才會被視為這台機器上的 worker，所以 GPU 伺服器不會被誤認為伺服器本機。它以自己的 worker token 驗證身分。

| 設定 | 預設值 | 說明 |
|---|---|---|
| `CODEFYUI_SSH_REMOTE_PORT_RANGE` | `18000-18100` | 通道可以在遠端主機 loopback 上使用的連接埠 |
| `CODEFYUI_SSH_STRICT_HOST_KEY_CHECKING` | `accept-new` | 設為 `yes` 時要求主機已知 |
