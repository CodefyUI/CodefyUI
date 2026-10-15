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

沒有 TLS 時，token 以及送到 worker 的每個 run 的輸入都會以明文經過網路；若發生在其他機器的連線上，伺服器會記錄警告。

## 連線中斷時 {/* #when-the-connection-drops */}

連線中斷的 worker 會繼續執行它的 run，並保留事件直到這台伺服器收到為止。在這台伺服器上，這些 run 顯示為 `disconnected`，並在 `CODEFYUI_WORKER_RESUME_GRACE_S`（5 分鐘）內保留名額；在此期間重新連線，run 會接續執行，紀錄沒有缺漏，超過時間則以 `interrupted` 結束。各種結果，以及筆電休眠的影響，見[連線中斷時](./run-queue#when-the-connection-drops)。
