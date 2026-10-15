---
sidebar_position: 3.7
title: 遠端 Worker
description: 在另一台機器上執行圖：用 Workers 面板產生的一次性 token，把 GPU 伺服器加入為 worker agent。
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
