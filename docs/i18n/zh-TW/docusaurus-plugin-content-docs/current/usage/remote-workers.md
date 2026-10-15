---
sidebar_position: 3.65
title: 遠端 Worker
description: 在另一台機器上執行圖——從 Workers 面板把 GPU 伺服器加入為 worker agent，透過直連、TLS proxy 或 SSH 連到控制面，共用或傳送檔案，保持版本與節點同步，信任，以及 agent 的 systemd 範例。
---

# 遠端 Worker

本頁帶你把一台剛安裝好的 GPU 伺服器，變成替你執行圖的 worker。若 worker 與伺服器在同一台機器上，看[多 GPU 執行](./multi-gpu#start-a-same-host-worker)就夠了；兩者背後的設計見[執行架構](/advanced/execution-architecture)。

## 概觀 {/* #overview */}

**Worker** 是在擁有運算資源的機器上執行的 `cdui agent start` 程序。它主動連到你的 CodefyUI 伺服器，也就是**控制面**，接收 target 指定它（或 `any`）的 run，並把每個 run 放在自己的子程序中執行，所以崩潰或記憶體不足被終止只會結束那一個 run。Worker 機器不開任何連接埠：agent 一律主動連到控制面，可以直連、經過 proxy，或經過控制面啟動的 SSH 通道。

有些東西從不離開控制面：

- 資料庫（run、事件、指標、artifact 資料列、已加入的 worker）：worker 從不開啟它；
- run 佇列與排程決策；
- 畫布：**Run** 一律在控制面執行，連同它的執行快取、持久化權重與 Inspector 輸出；
- session token：遠端 worker 以自己的憑證驗證身分。

送到 worker 的 run 帶著它的圖與選項，**其中 SECRET 參數的值已填回**；沒有共用資料夾時，也會帶上它使用的上傳檔（[儲存](#storage)）。請據此選擇連線方式（[信任](#trust-between-agent-and-control-plane)）。

Run 依它的 target 交給 worker：**Run** 旁的**排入佇列...**、`cdui run --target <名稱>`、`POST /api/runs` 的 `options.target`，或 sweep 與 study 的 `options.target`（[執行佇列：Workers](./run-queue#workers)）。

## 加入 worker {/* #adding-a-worker */}

1. **在 GPU 伺服器上安裝與控制面相同版本的 CodefyUI。** 先在控制面用 `cdui --version` 查看版本，然後在 GPU 伺服器上執行：

   ```bash
   curl -fsSL https://raw.githubusercontent.com/CodefyUI/CodefyUI/main/install.sh | CODEFYUI_RELEASE_TAG=3.0.2 bash
   ```

   請換成 `cdui --version` 印出的版本。版本不同的 agent 連線時會被拒絕（[讓 worker 保持同步](#keeping-workers-in-sync)）。需要特定 CUDA 或 ROCm 版本的 PyTorch 時，見 [GPU 與裝置設定](/getting-started/gpu-device)（`cdui install --gpu <選項>`）。
2. **讓 GPU 伺服器連得到控制面**，方式見[連到控制面](#reaching-the-control-plane)。使用 SSH 方式時，略過接下來兩步：控制面會自己啟動 agent。
3. **產生 token。** 開啟 **Workers**（**Settings → 執行 → Workers → 管理**，或 Runs 面板標題列的 **Workers**），按 **新增 worker**，停在 **另一台機器**。位址欄預設為此頁面使用的位址；若是 `127.0.0.1` 或 `localhost`，請改選伺服器的其他位址，或輸入 worker 連得到的位址。選擇 token 的有效期限（預設 24 小時）與可供幾個 agent 使用（預設 1），再按 **產生 token**。

   ![新增 worker 對話框與產生的加入指令](/img/add-worker-zh-TW.png)

4. **在 GPU 伺服器上執行指令**：

   ```bash
   cdui agent start --control http://192.168.1.5:8000 --token abcdef.0123456789abcdef
   ```

   Agent 會用 enrollment token 換取自己的 worker 憑證，存在 `<user-data>/agent/<host>-<port>.token`（只有該作業系統使用者可讀；安裝版的 `<user-data>` 是 `.codefyui_dev/`）。`--name` 指定 worker 名稱（預設為主機名稱），`--devices cuda:0,cuda:1` 限制它提供的 GPU，`--mount data=<path>` 指定共用資料夾（[儲存](#storage)）。所有參數見 [CLI 指令：Agent 指令](/getting-started/cli-commands#agent-commands)。
5. **確認它已上線。** 幾秒內 worker 就會以 **線上** 狀態出現在 Workers 面板，並顯示版本、節點型別與裝置。用**排入佇列...**或 `cdui run --target <名稱>` 送一個 run 給它。

   ![Workers 面板中一個線上的 worker](/img/workers-panel-zh-TW.png)

Enrollment token 只顯示一次，伺服器只保存它的雜湊值。已用完、已過期或已撤銷的 token 會被拒絕，訊息會說明是哪一種。已發出的 token 會列在對話框中，直到你刪除為止。

**重新啟動。** `cdui agent start --control <url>` 會找到已存的憑證並用它重新連線，不需要 token。伺服器重啟後也記得已加入的 worker：在它連線前列為 `offline`，指定它的 run 會在佇列中等待。`cdui agent forget <url>` 刪除已存的憑證，下次啟動時重新加入。

**移除 worker。** Workers 面板的 **移除** 會撤銷該 worker 的憑證、中斷它的連線（執行中的 run 立即以 `interrupted` 結束，包括正在等它重新連線的 run；佇列中的 run 失敗）並從清單移除；之後它的 agent 都會被拒絕。

**重新核發憑證。** 名稱一經加入即被占用：一般 token 無法再次加入該名稱（伺服器回 `409`）。要讓某個 worker 換新憑證（重灌機器、遺失憑證檔），請按它卡片上的 **重新核發**：在那裡產生的 token 只能加入該 worker，指令也會帶上它的 `--name`。使用後舊憑證即失效。若該機器上還留有舊憑證檔，請先在那裡執行 `cdui agent forget <url>`。

面板本身（狀態、排空、名額、被拒的連線）的說明見[多 GPU 執行：Workers 面板](./multi-gpu#the-workers-panel)。

## 連到控制面 {/* #reaching-the-control-plane */}

Agent 需要一個能連到控制面 HTTP 連接埠的網址：它以 `POST /api/workers/enroll` 加入，之後保持 `/ws/worker` WebSocket 連線。請從三種方式中擇一。

| 方式 | 適用情況 | 是否加密 |
| --- | --- | --- |
| [直連](#direct-lan-or-vpn) | 兩台機器在同一個可信任的區網或 VPN 上 | 否，除非 VPN 有加密 |
| [TLS proxy 後方](#behind-a-tls-proxy) | 控制面是其他人透過 HTTPS 連線的伺服器 | 是 |
| [透過 SSH](#connecting-over-ssh) | GPU 伺服器只能用 SSH 連到，或控制面是 NAT 後面的筆電 | 是（SSH 連線） |

### 直連：區網或 VPN {/* #direct-lan-or-vpn */}

`cdui start` 預設只監聽 `127.0.0.1`，其他機器無法連線。請以 `cdui start --host 0.0.0.0` 啟動。伺服器會接受自己各網路介面的位址（例如 `192.168.1.5:8000`）作為 `Host`；若 worker 改用主機名稱連線（`gpu-gw.lan:8000`、VPN 名稱），請在啟動伺服器前把它加入 `CODEFYUI_EXTRA_ALLOWED_HOSTS`：

```bash
CODEFYUI_EXTRA_ALLOWED_HOSTS=gpu-gw.lan:8000 cdui start --host 0.0.0.0
```

任何連得到該連接埠的人都能操作這台伺服器，worker token 與每個 run 的輸入也會以明文經過網路：請只在可信任的網路使用，例如實驗室區網，或 WireGuard、Tailscale 等 VPN。以未加密的 HTTP 連到另一台機器時，agent 會印出一次警告，Workers 面板也會把該 worker 標成 **未加密 HTTP**。

### TLS proxy 後方 {/* #behind-a-tls-proxy */}

若控制面要讓其他人透過網路連線，請依[部署在反向代理後方](./deployment)把它放在有 TLS 的 nginx 後面。那份站台設定已經涵蓋 `/ws/worker` 與 artifact 上傳（[這個 proxy 後方的 worker](./deployment#workers-behind-this-proxy)）。給 worker `https://` 位址：

```bash
cdui agent start --control https://codefyui.example.com --token abcdef.0123456789abcdef
```

Agent 每次連線都會檢查 proxy 的憑證（[信任](#trust-between-agent-and-control-plane)）；自簽憑證需要 `--ca-fingerprint`。

### 透過 SSH {/* #connecting-over-ssh */}

若 GPU 伺服器只能用 SSH 連到，例如筆電在 NAT 後面、伺服器只開放 22 埠，可以讓控制面自己啟動 worker。它會用 `ssh` 連到 GPU 伺服器、在那裡執行 `cdui agent start`，agent 再經由同一個 SSH 連線中的反向通道（`ssh -R`）連回來。兩台機器都不需要開連接埠，控制面也可以維持只監聽 `127.0.0.1`。

**開始之前：**

- GPU 伺服器上已用 `install.sh` 安裝相同版本的 CodefyUI（上面第 1 步）。控制面不會替你安裝。
- 控制面所在的機器 PATH 上有 OpenSSH 用戶端（`ssh`）。macOS 與 Linux 內建；Windows 請加入選用功能 **OpenSSH 用戶端**，並在 PATH 含有它的工作階段中啟動伺服器。
- 在控制面機器上、以執行 `cdui start` 的作業系統使用者開終端機執行 `ssh <target>`，**不會出現任何提示**就能登入。ssh 以 `BatchMode=yes` 執行：不能用密碼，有 passphrase 的金鑰須透過 `ssh-agent`。你的 `~/.ssh/config` 會生效，所以 `Host` 別名、經過跳板機的 `ProxyJump` 與其中指定的金鑰都能用：

  ```
  # ~/.ssh/config on the control plane's machine
  Host gpu-box
      HostName 10.0.3.17
      User alice
      ProxyJump bastion.lab
      IdentityFile ~/.ssh/id_ed25519
  ```

**新增。** 開啟 **Workers**，按 **新增 worker**，選 **透過 SSH**：

| 欄位 | 說明 |
|---|---|
| SSH 目標 | `user@host`，或伺服器帳號 `~/.ssh/config` 中的 `Host` 別名（上例的 `gpu-box`）。 |
| 連接埠 | ssh 的 `-p`。保持 22 時不會傳給 ssh，所以 ssh 設定中該主機的連接埠會生效。 |
| Worker 名稱 | 選填；預設為目標的主機名稱（`alice@gpu1.lab` 會成為 `gpu1`）。已被其他 worker 使用的名稱會被拒絕。 |
| 金鑰檔 | 選填；伺服器機器上的私鑰（ssh 的 `-i`）。留空則使用你的 ssh 設定與 agent。 |
| 遠端主機上的 cdui | 遠端啟動器的路徑；留空代表 `~/CodefyUI/cdui`，也就是 `install.sh` 放置的位置。 |

按 **新增並連線**。Worker 卡片上會多一行 **SSH 通道**：ssh 執行中但 agent 尚未連線時為 **連線中**，連上後為 **已連線**，等待重試時為 **失敗** 並附上 ssh 或 agent 最後輸出的幾行（金鑰被拒、未知主機、找不到 `cdui`），停用時為 **已停止**。**最近的 ssh 輸出** 可看到更多輸出。

**實際執行的內容。** 每個啟用中的 SSH worker，控制面都會維持一個像這樣的程序：

```bash
ssh -T -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
    -o StrictHostKeyChecking=accept-new \
    -R /home/alice/.codefyui-ssh/3f9a12c4-18000/sock:/tmp/cdui-ssh-k2j4/s -- alice@gpu-box \
    "sh -c '... ~/CodefyUI/cdui agent start --uds ~/.codefyui-ssh/3f9a12c4-18000/sock --foreground --control http://cp-3f9a12c4.invalid:18000 --name gpu-box'"
```

每次連線前，控制面會先另開一個短暫的 ssh，在遠端主機上建立 `~/.codefyui-ssh/<id>-<槽位>/`（權限 `0700`，`~/.codefyui-ssh` 也是），刪掉中斷的連線留下的 socket，並回報這個資料夾的完整路徑（`-R` 需要它）。轉送的遠端端點是這個資料夾裡的 Unix socket，所以兩台機器都需要 OpenSSH 6.7 以上，遠端 sshd 也必須允許 `AllowStreamLocalForwarding`（預設允許）；遠端主機不需要其他東西。`<id>` 是這個控制面的實例 ID（八個十六進位字元，存在它的使用者資料夾的 `ssh-instance-id`），所以在同一台 GPU 伺服器上使用同一個帳號的兩個控制面，不會共用資料夾、socket、憑證或 agent。轉送的本機端點（上例的 `/tmp/cdui-ssh-k2j4/s`）是控制面只為這個 worker 開的監聽端點：私有暫存資料夾裡的 Unix socket（Windows 上則是 `127.0.0.1` 的臨時連接埠），絕不是伺服器自己的連接埠。槽位編號取自 `CODEFYUI_SSH_REMOTE_PORT_RANGE`（`18000-18100`）並隨 worker 保存：它決定狀態資料夾，也是 agent 存放憑證所用的位址（`--control http://cp-<id>.invalid:<槽位>`，這個名稱不會解析到任何地方）；agent 連的是 socket（`--uds`）。遠端指令會先把 `~/.local/bin` 加到 PATH：非互動式的 ssh shell 不會讀取你的 shell 設定檔。ssh 結束後，控制面會在 2 秒後重新啟動它，間隔逐次加倍到最多一分鐘；控制面自己重啟後也會重新連線。若遠端 sshd 根本不允許轉送（`AllowStreamLocalForwarding no`、`DisableForwarding yes`；明確拒絕，或連續三次轉送失敗），通道會停在 **失敗** 並顯示原因，不再自動重試；請伺服器管理者允許轉送到 Unix socket 後按 **重試**。通道列也會顯示遠端 agent 的 PID：在遠端主機上 agent 的命令列是 `python .../scripts/dev.py agent start --uds ...`，可用 `pgrep -af 'agent start'` 找到它。

**憑證。** 每次啟動都會經由 ssh 的標準輸入送出一次性的 enrollment token（有效 10 分鐘），不會出現在任何命令列上，所以兩台機器的 `ps` 都看不到它。第一次啟動時 worker 會加入；agent 會把 worker 憑證以 `http://cp-<id>.invalid:<槽位>` 這個位址存在遠端主機上，之後每次啟動都用它重新連線，token 則不會被使用。Worker 加入後，token 會綁定到它，只能用來重新核發這個 worker 的憑證。這個 worker 第一次使用某個槽位時，會先對該位址執行 `cdui agent forget`，所以其他 worker 或其他控制面留下的憑證不會被送出。若仍送出了這種憑證（agent 回報控制面拒絕了它，或 agent 回報憑證屬於另一個 worker 名稱），控制面會先執行 `cdui agent forget` 再重新啟動 agent，讓它重新加入。所有通道都從控制面自己的位址連入，因此經由通道的加入嘗試會依 SSH worker 分開限速，彼此之間以及與那台機器上的 agent 互不影響。

**連線中斷。** 若連線不是因停用而結束（筆電睡眠、網路中斷），遠端 agent 與它的 run 會繼續執行；這些 run 在控制面上以 `disconnected` 等待並保留事件，如同[連線中斷時](#when-the-connection-drops)所述。控制面會以相同的槽位重新啟動 ssh，轉送會回到同一個 socket 路徑，遠端腳本發現 agent 仍在執行，就會接上它而不另啟一個（通道列會顯示重新接上的 PID）。Agent 會自行經由新的通道重新連線，它的 run 也會繼續。若在 `CODEFYUI_WORKER_RESUME_GRACE_S`（5 分鐘）內沒有新的連線接上，agent 會停止。遠端腳本把 agent 的 PID 與日誌存在遠端主機的 `~/.codefyui-ssh/<id>-<槽位>/`，與 socket 放在一起。

**停用、啟用、刪除。** **停用** 會先讓 worker 停止接新 run，再結束它的 ssh 連線；之後即使有 agent 以它的名稱連線，停用中的 worker 仍維持不接新 run。停用會經由連線通知遠端 agent 停止（沒有連線時則另開一個短暫的 ssh），agent 會停止仍在執行的 run；若有需要，請先排空並等它完成。Agent 會要求每個 run 停止並給它 10 秒寬限：在寬限內停下的 run 以 `cancelled` 結束，超過寬限而被它終止的 run 以 `interrupted` 結束。Agent 會上傳 transfer run 寫出的檔案（它的中斷檢查點），回報每個 run 如何結束後才退出；控制面最多等它 60 秒再結束 ssh，屆時仍未回報的 run 會立即以 `interrupted` 結束，不必等 `CODEFYUI_WORKER_RESUME_GRACE_S`。移除也以同樣方式停止它。伺服器關閉時，會在伺服器自己的 run 收尾的同時停止所有 SSH worker（不是先停 SSH worker），SSH worker 總共最多 25 秒。`cdui stop` 約 2 秒後就會強制結束伺服器：agent 已先收到停止通知，ssh 連線會在它們結束後自行結束，但屆時尚未回報的 run 會在下次啟動時標為 `interrupted`。舊版留在 `~/.codefyui-ssh/<槽位>/` 舊配置、以 TCP 連接埠連線的遠端 agent，會在下一次連線的準備步驟中，確認沒有連線接著它之後被停止（仍有連線接著時只回報）。**啟用** 會重新啟動連線。**刪除**（尚未連線的 worker）與 **移除**（已連線的 worker）效果相同：結束連線、刪除 SSH 項目，並如[移除 worker](#adding-a-worker) 所述撤銷該 worker，所以之後可以用同一個名稱再新增。兩者也會清理遠端主機，停用則不會（啟用會沿用已存的憑證）：回應之後，控制面在背景再開一個短暫的 ssh，若 agent 仍在執行就先停止它，再刪除它為 `http://cp-<id>.invalid:<槽位>` 存的憑證（`cdui agent forget`），並刪除 `~/.codefyui-ssh/<id>-<槽位>/`。只會動到這個控制面自己的槽位，不會碰其他控制面的；清理完成前，其他 SSH worker 也不會使用這個槽位。若仍有連線接在那裡，或 agent 停不下來，兩者都會保留。若遠端主機約一分鐘內沒有回應，伺服器日誌會列出留下的東西；要手動清除，請在遠端主機執行 `cdui agent forget http://cp-<id>.invalid:<槽位>` 與 `rm -r ~/.codefyui-ssh/<id>-<槽位>`。

**安全性。**

- 遠端主機上只有你的帳號能連到通道。它的遠端端點是 `~/.codefyui-ssh/<id>-<槽位>/` 裡的 Unix socket，該資料夾權限為 `0700`，所以共用 GPU 伺服器的其他使用者既無法連上它，也無法搶先占用它的位址來攔截 agent 的 token；若用遠端 loopback 上的 TCP 連接埠，這兩件事都做得到。
- 在控制面上，通道接到它專用的監聽端點：每個 SSH worker 各開一個，絕不是伺服器的連接埠；它是私有（`0700`）暫存資料夾裡的 Unix socket，所以控制面那台機器的其他使用者也連不到（Windows 上則是 `127.0.0.1` 的臨時連接埠，他們連得到）。在那裡被拒的 token 不會觸發任何處理：只有 agent 自己的回報才會讓控制面清除它的憑證。這個監聽埠只回應 agent 會用到的路徑：帶 worker token 的 `/ws/worker`（session token 在那裡一律被拒）、帶 enrollment token 的 `POST /api/workers/enroll`，以及在帶有有效 worker token 時，transfer 後端使用的路徑（[儲存：沒有共用檔案系統](#storage-no-shared-filesystem)）：三個輸入下載路徑（`GET /api/files/download/...`、`/api/images/download/...`、`/api/models/download/...`）與 `POST /api/runs/{id}/artifacts/upload`。這些下載路徑在區網上是公開讀取，但經由通道時沒有 worker token 會回 `403`。其他路徑一律回 `404`，包括 `GET /api/auth/bootstrap` 與網頁介面，不論 request 帶的是哪個 `Host`。在伺服器自己的連接埠上，沒有任何 `Host` 會讓 request 被當成來自通道：SSH worker 的位址在那裡只是不在 allowlist 內的 `Host`（`421`）。
- `StrictHostKeyChecking=accept-new` 會在第一次見到主機時信任它，金鑰改變時拒絕連線。設定 `CODEFYUI_SSH_STRICT_HOST_KEY_CHECKING=yes` 則要求主機已在伺服器帳號的 `known_hosts` 中。
- 控制面使用執行它的作業系統使用者的 SSH 存取權，就像 Source Control 分頁使用 git 一樣。能操作這台伺服器的人，就能在該使用者連得到的每台主機上啟動 SSH worker。
- 通道後面的 agent 從 `127.0.0.1` 連線，但只有在它的 machine id 與伺服器相同時才會被視為控制面機器上的 worker，所以 GPU 伺服器不會被誤認為伺服器本機。它以自己的 worker token 驗證身分。

### 控制面放在哪裡 {/* #where-to-run-the-control-plane */}

控制面保管佇列並記錄每個事件，所以 worker 上的 run 仰賴它持續運作、連得到。

- **筆電可以用，但有限制。** 蓋上螢幕會關閉所有 worker 連線。worker 未在 `CODEFYUI_WORKER_RESUME_GRACE_S`（5 分鐘）內回來的 run 會以 `interrupted` 結束（[連線中斷時](#when-the-connection-drops)）；筆電換到另一個網路，直連的 worker 要連的位址也會改變（SSH worker 會跟著走，因為通道是從筆電發起）。短暫休眠可調高 `CODEFYUI_WORKER_RESUME_GRACE_S`；過夜的訓練請讓筆電保持清醒，或使用常駐的控制面。
- **常駐伺服器是穩健的選擇。** 在一台持續運作的機器（GPU 伺服器本身，或旁邊的小主機）上以 systemd 服務執行 `cdui start`（[部署：systemd unit](./deployment#a-systemd-unit)），從筆電使用編輯器。控制面維持在 `127.0.0.1` 時，瀏覽器透過 SSH 連接埠轉送即可，其他什麼都不必對外開放：

  ```bash
  ssh -N -L 8000:127.0.0.1:8000 gpu-box      # then open http://localhost:8000
  ```

  同一台機器上的 worker 用同機指令；其他地方的 worker 依上述方式直連、經過 proxy，或透過 SSH。

## 儲存 {/* #storage */}

不使用資料夾中檔案的 run（純 tensor 的圖，或 Dataset、HuggingFaceDataset 節點下載到自己快取的圖）可以在任何 worker 上執行。會讀取上傳檔或寫出檔案的 run，需要兩者之一：兩台機器共用的資料夾，或經由 agent 連線搬運檔案的 transfer 後端。有共用資料夾的 worker 優先。

| Run | 有 `data` 掛載的 worker | 沒有掛載的 worker |
| --- | --- | --- |
| 不使用檔案 | 可排入 | 可排入 |
| 使用檔案，且全部是在檔案欄位中選取的上傳檔 | 可排入（優先） | 可排入：檔案經由 HTTP 傳送（[沒有共用檔案系統](#storage-no-shared-filesystem)）。`any` 會在有掛載的 worker 之後才試它 |
| 以其他方式使用檔案（一般路徑、這台伺服器沒有的上傳檔、輸入超過 `CODEFYUI_TRANSFER_MAX_INPUT_BYTES`），或 `CODEFYUI_TRANSFER_ENABLED=0` | 可排入 | 指定該 worker：送出時 `400`，``worker gpu-box-1 has no `data` mount; add `--mount data=...` to its agent, or make the run fit the transfer backend: <reason>``。`any`：略過 |

### 共用檔案系統 {/* #storage-shared-filesystem */}

Run 有以下節點時就算使用檔案：讀取上傳檔的節點（CSVReader、ImageReader、ModelLoader、CheckpointLoader 等上傳欄位）、會存檔的節點（CheckpointSaver、ModelSaver、ImageWriter、VideoWrite）、設定了 `checkpoint_every`、`checkpoint_every_steps` 或 `tensorboard` 的 TrainingLoop，或路徑指向資料夾的讀取節點。在兩台機器掛上同一個網路共用資料夾，並給它相同的 **label**：控制面的資料夾（使用 `--project` 時為專案目錄）的 label 是 `data`（`CODEFYUI_MOUNT_LABEL`），agent 以 `cdui agent start --mount data=<path>` 指定自己這邊的路徑。兩台機器的路徑可以不同。

NFS 範例，兩台機器的掛載路徑不同：

```bash
# Both machines
sudo mount -t nfs nas.lab:/export/codefyui /mnt/codefyui      # on the GPU server: /nas/codefyui
# Control plane: keep the data folders on the share (the database stays local)
export CODEFYUI_DATA_FILES_DIR=/mnt/codefyui/data/files CODEFYUI_IMAGES_DIR=/mnt/codefyui/data/images \
       CODEFYUI_MODELS_DIR=/mnt/codefyui/data/models CODEFYUI_MEDIA_DIR=/mnt/codefyui/data/media
cdui start
# GPU server: the label is saved, so later starts need no --mount
cdui agent start --mount data=/nas/codefyui/data
```

在 worker 上，run 的資料夾依 `backend/data` 的結構由掛載路徑推得（`files/`、`images/`、`models/`、`media/`、`graphs/`、`presets/`）；路徑中有 `codefyui.project.toml` 時則視為該專案。Run 記錄的檔案（checkpoint、TensorBoard 資料夾）以相對於掛載點的路徑儲存（`backend: "mount"`），`GET /api/runs/{id}/artifacts` 與執行任務面板列出的是控制面上的路徑；控制面沒有該掛載的資料列 `path` 為 `null`，以 label 與相對路徑表示。與伺服器在同一台機器上的 agent 不需要 `--mount` 就擁有 `data`：它的 run 直接使用伺服器設定的資料夾，產出的檔案列為一般的本機檔案，`GET /api/workers` 會以伺服器的路徑顯示該 label。

掛載設定存在使用者資料目錄的 `agent.json`，之後啟動會沿用；`--mount data=` 會移除一個。Agent 啟動時已儲存的掛載資料夾若不存在（共用資料夾尚未掛上），在資料夾出現前不會提供該 label；agent 每次心跳都會重新檢查。Agent 執行中不能變更 `--mount`，請先停止它。使用 `--project` 時，上傳的資料檔仍在安裝目錄的 `data/files`，不在專案內，因此不會共用。

### 沒有共用檔案系統 {/* #storage-no-shared-filesystem */}

會讀取上傳檔或寫出檔案的 run，也可以交給與控制面沒有任何共用資料夾的 worker。此時控制面就是檔案庫：把 run 用到的上傳檔送給 worker，並接收 run 寫出的檔案。只有 run 指定的 worker 沒有共用資料夾，或 `any` 找不到有空的此類 worker 時，才會走這條路。

**輸入。** 能用這種方式送過去的，是在檔案欄位中選取的上傳檔：CSVReader 等資料檔欄位、ImageReader、ModelLoader、CheckpointLoader。指派時，控制面以 SHA-256 雜湊值標示每個檔案。Agent 把下載過的檔案存在 `<user-data>/agent-cache/<sha256>`，之後使用同一個檔案的 run 不會再下載；快取中沒有的檔案，以 worker 的 token 從控制面既有的下載路徑取得，並核對雜湊值。下載期間，執行任務面板的紀錄會顯示 `fetching inputs 2/3: train.csv (1.2 MiB, fetched)`（kind 為 `transfer_progress` 的 `run_warning`）；顯示 `cached` 而非 `fetched` 表示取自快取。下載期間按停止，run 以 `cancelled` 結束。

**Run 的工作資料夾。** 每個 run 在 worker 上有自己的資料夾 `<user-data>/agent-runs/<run id>/data`，結構與 `backend/data` 相同（`files/`、`images/`、`models/`、`media/`），每個輸入檔以圖中使用的名稱複製一份放在裡面（節點覆寫輸入檔也碰不到快取）。Worker 自己的資料夾不會被動到，只有 `cache/` 資料夾（LMTokenizedDataset 的 token 區塊）由各 run 共用。Dataset、HuggingFaceDataset 下載的資料集仍放在 worker 自己的快取，與在伺服器上相同。

**輸出。** 在 run 的結束訊息送到控制面之前，agent 會上傳：

- run 記錄為 artifact 的每個檔案（定期 checkpoint、TensorBoard 資料夾則逐檔上傳），在 run 回報時上傳到 `<資料夾>/runs/<run id>/<kind>/<name>`；
- run 結束後，它在 `models/`、`media/`、`output/` 底下寫出的其他檔案（CheckpointSaver、ModelSaver、VideoWrite、ImageWriter），以及被它覆寫的輸入檔，上傳到控制面自己的資料夾、相同的相對路徑，也就是同一個節點在本機 run 時寫入的位置（`CODEFYUI_MODELS_DIR` 中的 `models/best.pt`）。該處已有不同的檔案時會保留原檔，新檔加上後綴（`best-1.pt`）；內容相同則不重複存放。這些檔案的 kind 為 `output`；
- 最後是被停止的訓練迴圈寫出的 checkpoint，存到 run 的資料夾，讓已取消 run 的續訓點留在控制面上。

上傳與 run 並行，網路慢不會拖住 run；上傳完成後才回報 run 結束。`GET /api/runs/{id}/artifacts` 以 `backend: "transfer"` 和控制面上的路徑列出，執行任務面板在每一筆旁提供 **下載**（`GET /api/runs/{id}/artifacts/{artifact_id}/download`）。連不上控制面時（連線中斷，或 proxy 回應 502、503、504），agent 會持續重試，間隔逐次拉長，時間與控制面保留已斷線 worker 之 run 的時間相同（`CODEFYUI_WORKER_RESUME_GRACE_S`，預設 5 分鐘），所以在斷線期間結束的 run 仍會送回它的檔案。同一個 run 的檔案共用這段時間：用完之後，其餘每個檔案只再試一次。超過控制面上傳上限（`CODEFYUI_MAX_UPLOAD_SIZE`，500 MB）的檔案，或重試後仍未被接收的檔案，會留在 worker 上，以它在 worker 上的路徑列出，並附上 `transfer_failed` 警告。在最後一批上傳期間按停止，尚未送出的檔案就不再上傳，並以一則 `transfer_failed` 警告說明有幾個留在 worker 上。

**清理。** Run 自己的資料夾屬於該 run：run 保留機制（`CODEFYUI_RUN_RETENTION_KEEP_LAST`）或 **刪除** 移除 run 時，它的 `runs/<run id>` 資料夾也一起刪除。想保留的檔案請下載或另行複製。以 `interrupted` 結束、之後被保留機制移除的 run 會保留 `checkpoint` 資料夾，與本機 run 保留續訓點的做法相同。放進資料夾的檔案屬於你，與本機 run 相同：兩者都不會刪除它們。在 worker 上，run 的資料夾跟著 agent 自己的 run 保留機制；快取不會自動清理，刪除 `agent-cache` 是安全的（需要時會重新下載）。

無法以此方式處理的 run，若指定了沒有共用資料夾的 worker，會在送出時被拒絕，訊息會說明原因：讀取節點用的是一般路徑而非上傳檔（FileReader、ImageFolderDataset、ImageBatchReader、VideoLoad、讀取資料夾的 DocumentLoader）、控制面上沒有該上傳檔，或輸入超過上限。例如：``worker gpu-box-1 has no `data` mount; add `--mount data=...` to its agent, or make the run fit the transfer backend: its inputs add up to 3.10 GiB, over the transfer limit of 2.00 GiB (CODEFYUI_TRANSFER_MAX_INPUT_BYTES)``。

輸入與輸出送往 agent 連線的同一個位址，帶著它的 worker token，並做相同的憑證檢查（釘選的憑證對它們同樣生效）：沒有 TLS 時，它們會以明文經過網路。在 proxy 後方時，它的請求大小上限必須容得下最大的輸出（[部署](./deployment#workers-behind-this-proxy)）。

## 讓 worker 保持同步 {/* #keeping-workers-in-sync */}

Worker 必須執行與控制面相同的 CodefyUI 版本，也應該認得相同的節點。Agent 每次連線時，控制面都會檢查這兩項。

**版本。** 版本不同的 agent 會被拒絕，原因中列出雙方版本，例如 `version mismatch: worker 3.1.0, control plane 3.1.2`。Agent 會停止而不重試，Workers 面板則在 **被拒的連線** 下列出該名稱與原因，所以一直沒出現的 worker 會說明原因；試圖搶用已連線 worker 名稱的 agent 會連同位址列在那裡，不會顯示在該 worker 的卡片上。版本會在 agent 第一則訊息中最先被讀取，所以即使訊息格式已改變的 agent 也會得到這個原因。要更新時，請在控制面與每個 worker 上執行 `cdui update`（或安裝相同的 `CODEFYUI_RELEASE_TAG`），再重新啟動 agent；該名稱連線成功後，拒絕紀錄就會清除。面板會保留每個名稱最近一次的拒絕，直到伺服器重新啟動。

**節點型別。** 外掛、Package Center 套件與自訂節點都是每台機器各自安裝。Agent 啟動時，一個短暫的輔助程序會以 run 的方式載入所有節點，回報節點型別與其定義的雜湊（節點目錄）。雜湊與控制面不同時，worker 會標為 **Degraded**，卡片列出 worker 缺少的型別，以及只有它有的型別。降級的 worker：

- 缺少控制面有的型別時，永遠不會收到 target 為 `any` 的 run（只多出型別的 worker 仍可接收）；
- 只有在圖用到的每個節點型別它都有時，才會收到 run：包括展開 preset 與 subgraph 之後的節點，以及 **Map** 節點所執行的每個已安裝 preset 中的節點。否則送出時就會被拒絕並列出缺少的型別；佇列中的 run 在送出前也會以同樣方式失敗，以防 worker 在等待期間以較少的型別重新連線。

要解除降級，請在 worker 上安裝相同的外掛與套件（`cdui plugin install ...`、Package Center、相同的自訂節點檔）並重新啟動 agent：它只在啟動時回報節點。輔助程序失敗的 worker 不會回報節點目錄：它的卡片會顯示警告，`any` 會略過它，指定它的 run 則不經檢查直接送出（agent 的 log 會說明輔助程序失敗的原因）。目前還沒有把控制面的外掛組合複製到 worker 的指令。

**環境指紋。** 同一個輔助程序也會收集 worker 的[環境指紋](/advanced/api-reference#run-environment)（CodefyUI commit、Python、torch build、加速器、外掛版本、影響數值的環境變數）。Workers 面板顯示它的摘要，worker 執行的每個 run 都會記錄這個摘要（重新連線時未回報指紋的 worker 會顯示「未回報」，不會沿用舊的摘要），所以 Runs 面板會指出實際執行它的環境。指紋描述的是 agent 的環境，也就是它的 run 子程序所繼承的環境，在 run 的裝置集合縮小 `CUDA_VISIBLE_DEVICES` 之前。

## 連線中斷時 {/* #when-the-connection-drops */}

Worker 的連線關閉，或連續錯過三次心跳（30 秒）時，即視為斷線。它的 run 會在 worker 上繼續執行，worker 把事件存在磁碟上，直到控制面確認已經記錄。每個 run 的狀態變成 `disconnected`，並留下類型為 `worker_disconnected` 的 `run_warning`；名額仍然保留，執行任務面板會顯示 worker 已斷線多久（`GET /api/runs/{id}` 以 `worker_status` 回報）。接下來會是三種情形之一：

- **Worker 在寬限時間內重新連線**（`CODEFYUI_WORKER_RESUME_GRACE_S`，5 分鐘）：它依序補送控制面尚未記錄的事件，run 回到 `running`，並留下類型為 `worker_reconnected` 的 `run_warning`。事件紀錄沒有缺漏，也沒有重複。
- **沒有重新連線**：run 以 `interrupted` 結束，名額釋放。如果 worker 之後回來且 run 仍在執行，會被要求立即停止，讓這張卡留給下一個 run；run 在 worker 上的結束方式會記錄為 run 的 `late_result`（`{status, error, worker_id, at}`，見 `GET /api/runs/{id}` 與 `execution_facts`），狀態與事件紀錄維持原樣。
- **控制面在這段期間重新啟動**：run 會在啟動時以 `interrupted` 結束，與其他執行中的 run 相同。

`CODEFYUI_WORKER_RESUME_GRACE_S` 由控制面讀取；設為 `0` 則 worker 一斷線就結束它的 run。SSH worker 的 agent 也以同樣的寬限時間決定何時不再等待通道（[連線中斷](#connecting-over-ssh)）。

## Agent 與控制面之間的信任 {/* #trust-between-agent-and-control-plane */}

送到 worker 的 run 會帶著填回 secret 參數值（API 金鑰、密碼）的圖，所以兩端都必須確認對方是誰。控制面以 worker 憑證認出 agent；agent 則以 TLS 憑證認出控制面，加入請求、每一條 `/ws/worker` 連線與每一次檔案傳輸都做同一個檢查。這對共用實例上的 SECRET 參數代表什麼，見[共用實例：送到 worker 的執行](./shared-instances#runs-sent-to-a-worker)。

CodefyUI 本身不提供 TLS；請把它放在 TLS proxy 後面（[TLS proxy 後方](#behind-a-tls-proxy)），並給 worker `https://` 位址，或使用 [SSH](#connecting-over-ssh) 方式，由 SSH 連線加密通道。

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

位址是 `https://` 時，新增 worker 對話框會顯示這個指令。若控制面自己讀得到憑證檔，把 `CODEFYUI_TLS_CERT_FILE` 設為它的路徑：對話框就會顯示指紋，產生的指令也會帶上 `--ca-fingerprint`（伺服器只在檔案改變時重新讀取）。Worker 的 CA 信任該憑證時請不要設定，以免指令釘選一把續期時就會換掉的金鑰。

**Worker 的 CA 不接受、又沒有釘選的憑證**不會被使用：加入會失敗；執行中的 agent 會記錄錯誤與原因（第一次會附上伺服器出示的指紋，供你與對話框比對後用 `--ca-fingerprint` 帶入），並以平常的退避間隔重試，所以修好 proxy 或更新 CA 清單後不必重啟。

**已釘選的憑證換了金鑰之後**，先在 proxy 主機上確認新指紋，然後擇一：

- 執行 `cdui agent start --control <url> --ca-fingerprint <新指紋>`：保留 worker 憑證，連線成功後記錄新的釘選值；或
- 執行 `cdui agent forget <url>`：刪除憑證與其中的釘選值，再用新的 token 重新加入。

續期的憑證只有在沿用金鑰時（certbot 的 `--reuse-key`）指紋才不變。

**未加密的 HTTP。** 以 `http://` 位址連到另一台機器時，`cdui agent start` 會印出一次警告：worker token 與每個指派中的 secret 參數都以明文傳輸。Agent 仍會連線。Workers 面板會把這種 worker 標成 **未加密 HTTP**：它的連線從另一台機器未加密地抵達伺服器。伺服器依連線的 scheme（只有 uvicorn 回報 `wss` 才算，uvicorn 只對 `--forwarded-allow-ips` 列出的 proxy 這麼做；CodefyUI 本身從不讀 `X-Forwarded-Proto`）與對端位址判斷：loopback 對端，例如同一台機器的 agent 或 SSH 通道的本機端，不會被標記，因為跨越網路的不是那條 socket。同樣的理由，`http://127.0.0.1` 位址不會讓 agent 印出警告。

**在 worker 上。** Agent 會在記憶體中把 run 的 secret 值交給它的子程序，自己不會寫入任何檔案，但 worker 機器在 run 執行期間持有它們：管理那台機器的人等同被託付這些值。Worker 憑證檔只有 agent 的作業系統使用者可讀。

**在控制面上。** 已加入的 worker 可以把檔案寫進控制面的 `models/`、`media/` 與 `output/` 資料夾，並記錄為它的 run 可下載的 artifact（[沒有共用檔案系統](#storage-no-shared-filesystem)），所以只加入你信任的機器。

不涵蓋：mutual TLS（agent 由 worker 憑證識別）、由 CodefyUI 管理的憑證機構，以及使用者身分。

## 以服務執行 agent {/* #running-the-agent-as-a-service */}

在 Linux GPU 伺服器上，請以 systemd 執行 agent，讓它開機時啟動、崩潰後自動重啟。這與[部署：systemd unit](./deployment#a-systemd-unit) 中伺服器的 unit 對應：安裝在 `/opt/codefyui`，由 `codefyui` 使用者擁有。

**先以服務使用者加入一次**，讓憑證存進該使用者的資料目錄，再停止指令啟動的背景 agent：

```bash
sudo -u codefyui /opt/codefyui/cdui agent start --control https://codefyui.example.com --token abcdef.0123456789abcdef
sudo -u codefyui /opt/codefyui/cdui agent stop
```

需要時在這裡加上 `--name`、`--mount` 與 `--ca-fingerprint`：名稱與釘選值存在憑證檔，掛載存在 `agent.json`，所以 unit 不必重複。

```ini
[Unit]
Description=CodefyUI worker agent
After=network-online.target
Wants=network-online.target
# On an NFS-backed --mount, also: After=remote-fs.target

[Service]
Type=exec
User=codefyui
Group=codefyui
WorkingDirectory=/opt/codefyui

# --foreground on purpose: systemd is the supervisor, so the agent must not
# daemonize away from it. The stored credential (from the enrollment above)
# is used; no token is needed here.
ExecStart=/opt/codefyui/cdui agent start --foreground --control https://codefyui.example.com

Restart=on-failure
RestartSec=5s
# On SIGTERM the agent asks its runs to stop and terminates them after 10 s.
TimeoutStopSec=60s
# SIGTERM to the agent only: with the default (control-group) systemd also
# signals the run subprocesses directly, and a running run ends `failed`
# instead of `cancelled`. The rest of the group is killed after TimeoutStopSec.
KillMode=mixed

# Deliberately NOT PrivateDevices (breaks GPU access) or ProtectHome /
# ProtectSystem=strict (both break writes to the user data dir under
# /opt/codefyui/.codefyui_dev/, where the credential, spool and run folders live).
NoNewPrivileges=yes
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
```

```bash
sudo install -m 644 codefyui-agent.service /etc/systemd/system/codefyui-agent.service
sudo systemctl daemon-reload
sudo systemctl enable --now codefyui-agent
journalctl -u codefyui-agent -f
```

`--foreground` 是必要的，理由與伺服器相同：沒有它時，`cdui agent start` 會啟動背景 agent 後就結束，而 systemd 會把 `Type=exec` 服務主程序的結束視為服務結束。因版本不同而被拒的 agent 會以錯誤結束，`Restart=on-failure` 會每 5 秒重試一次直到你更新；更新期間請先 `systemctl stop codefyui-agent`。請保留 `KillMode=mixed`：它讓 agent 自己停止它的 run，執行中的 run 會以 `cancelled` 結束（訓練迴圈會寫下中斷 checkpoint）；使用 systemd 的預設值時，run 子程序會直接收到 SIGTERM，run 會以 `failed` 結束，錯誤為 `run subprocess exited with code -15`。SSH worker 不需要 unit：控制面會讓它的 agent 持續執行。

## 限制 {/* #limits */}

- **畫布在控制面執行。** **Run** 一律在那裡執行，連同它的快取與持久化權重；要把分頁的圖送到 worker，請使用**排入佇列...**。
- **Worker 上不擷取輸出。** 目標為 worker 的 run 不接受 `record_outputs`（[#789](https://github.com/CodefyUI/CodefyUI/issues/789)）；要為 Inspector 擷取輸出，請以控制面為目標。Worker run 每次也都從新的權重開始。
- **沒有依記憶體的准入。** Run 與在控制面上一樣，以每個裝置名額的數量准入；沒有任何機制檢查 run 是否放得進卡的記憶體（[#788](https://github.com/CodefyUI/CodefyUI/issues/788)）。
- **一個 run，一台機器。** Run 只使用一個 worker 的裝置；沒有跨卡或跨機器的 `DistributedDataParallel`（[#792](https://github.com/CodefyUI/CodefyUI/issues/792)）。
- **以 Linux 為主。** Worker 在 Linux 上開發與測試；macOS 可用，Windows 盡力支援。
- **控制面重新啟動會結束 worker 上的 run。** 它停止時正在執行的 run 會以 `interrupted` 結束，與本機 run 相同（[執行佇列：伺服器停止時](./run-queue#when-the-server-stops)）。

## 設定 {/* #settings */}

在控制面上：

| 設定 | 預設值 | 說明 |
| --- | --- | --- |
| `CODEFYUI_EXTRA_ALLOWED_HOSTS` | *（空）* | Worker（或瀏覽器）連線用的主機名稱，例如 `gpu-gw.lan:8000` |
| `CODEFYUI_WORKER_RESUME_GRACE_S` | `300` | 斷線 worker 的 run 等待它的秒數；`0` 立即結束 |
| `CODEFYUI_TLS_CERT_FILE` | *（空）* | Proxy 的憑證（PEM，leaf 在前），讓新增 worker 對話框顯示它的指紋 |
| `CODEFYUI_MOUNT_LABEL` | `data` | 控制面資料夾的 label |
| `CODEFYUI_TRANSFER_ENABLED` | `1` | `0` 關閉 transfer 後端 |
| `CODEFYUI_TRANSFER_MAX_INPUT_BYTES` | 2 GiB | 一個 run 經由傳送的輸入檔總大小上限 |
| `CODEFYUI_MAX_UPLOAD_SIZE` | 500 MB | Worker 可上傳的單一輸出檔上限；所有檔案上傳都適用同一上限 |
| `CODEFYUI_SSH_REMOTE_PORT_RANGE` | `18000-18100` | SSH worker 的槽位編號：決定 worker 在遠端主機上的狀態資料夾（`~/.codefyui-ssh/<id>-<槽位>/`，通道的 socket 就在裡面），也是它存放憑證所用的位址 |
| `CODEFYUI_SSH_STRICT_HOST_KEY_CHECKING` | `accept-new` | 設為 `yes` 時要求主機已在 `known_hosts` 中 |

在 worker 上，未給 `--token` 時由 `CODEFYUI_AGENT_TOKEN` 提供 token（[CLI 指令：環境變數](/getting-started/cli-commands#environment-variables)）。

## 另請參閱 {/* #see-also */}

- [多 GPU 執行](./multi-gpu)——伺服器本機上的 worker、裝置集合、名額與 Workers 面板
- [執行佇列：Workers](./run-queue#workers)——`target` 如何放置 run
- [部署](./deployment)——TLS proxy 與伺服器的 systemd unit
- [執行架構](/advanced/execution-architecture)——worker 協定、不變量與設計決策
- [API 參考](/advanced/api-reference)——`/api/workers` 端點與 worker token
