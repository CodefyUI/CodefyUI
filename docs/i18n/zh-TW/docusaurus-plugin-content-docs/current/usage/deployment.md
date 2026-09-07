---
sidebar_position: 7.65
title: 放在反向代理後面
description: 給 CodefyUI 一個自己的網域名稱、只綁定回送位址，再用 nginx 在前面處理 TLS 與單一登入 -- 附上實際測試過的 systemd 單元與 nginx 設定。
---

# 放在反向代理後面

CodefyUI 是透過 HTTP 提供服務的桌面工具，沒有使用者帳號、登入或個人歸屬，後端也無法指出一次執行是由誰發起。實務影響請見[共用的伺服器](./shared-instances)。團隊若要在共用伺服器前加入身分驗證，應使用已整合組織身分系統的反向代理，並讓 CodefyUI 只綁定代理後方的回送位址。

以下設定已使用實際的 nginx 與 systemd 測試，限制會直接標示。

```text
                       TLS + 你們的單一登入            純 HTTP，只走回送位址
  瀏覽器  ────────────────────────────────►  nginx  ──────────────────────────►  cdui start
           https://codefyui.example.com        :443                              127.0.0.1:8000
```

必須同時符合以下三項，其中第二項最常被忽略。

1. CodefyUI 要有自己的**網域名稱**，不能掛在子路徑底下。
2. 那個名稱必須加進 CodefyUI 的 **Host 白名單**，否則每一個請求都會失敗。
3. CodefyUI 只綁定**回送位址**，讓代理是唯一的入口。

## 1. 給它一個網域名稱，不要用子路徑

`https://codefyui.example.com/` 可以。`https://tools.example.com/codefyui/` 不行。

build 出來的前端產生的是根目錄絕對路徑：`index.html` 連的是 `/assets/index-*.js`，後端也是掛在字面上的 `/assets`，API 客戶端的基底路徑則是字面上的 `/api`。這裡沒有 `base` 設定，也沒有任何 build 期的環境變數可以改，所以要掛在子路徑就得自己 fork 一份前端重新 build。

:::note 為什麼目前不支援子路徑
這類應用程式通常使用獨立的 vhost，只需要新增一筆 DNS 紀錄。支援子路徑則必須讓同一項 build 設定貫穿前端、API 客戶端與後端靜態掛載，而多數團隊不需要這種部署方式。如果確實無法配置主機名稱，請開 issue 說明原因；這類使用情境可能改變此決定。
:::

## 2. 把對外的主機名稱加進白名單

**若漏掉這一步，瀏覽器會顯示空白頁面。**

CodefyUI 會檢查每個請求的 `Host` 標頭，凡是不認得的一律回 `421 Misdirected Request`：

```json
{"detail": "Misdirected Request (Host not allowed)"}
```

這項檢查用來防止 DNS rebinding，而且位於最外層，因此會套用到**網頁本身**，不只套用到 API。代理後方收到的 `Host` 是公開主機名稱，預設不在白名單內；瀏覽器因此收到 JSON 而不是 `index.html`，無法載入介面。由於介面尚未載入，畫面上也不會顯示 UI 錯誤訊息。

用一個環境變數解決：

```bash
export CODEFYUI_EXTRA_ALLOWED_HOSTS="codefyui.example.com"
```

| 規則 | 細節 |
| --- | --- |
| 分隔符號 | 逗號。前後空白會被去掉，`a, b` 沒問題。 |
| 比對方式 | 完整字串比對，不分大小寫。**不支援萬用字元** -- `*.example.com` 永遠不會相符。 |
| 埠號 | 是字串的一部分。瀏覽器網址列上的主機加埠號是什麼，就寫什麼。 |
| 生效時機 | 只在啟動時讀取，而且不會被保存下來 -- 每次啟動伺服器都要設。 |

:::warning 測試環境也必須注意埠號
在 `:443` 或 `:80` 上，瀏覽器送的是不帶埠號的 `Host: codefyui.example.com`，所以白名單要寫沒有埠號的名稱。在其他埠上，它會送出 `Host: codefyui.example.com:8443`，白名單就必須包含**帶埠號的完整值**。只寫 `codefyui.example.com` 不會相符，寫 `codefyui.example.com:443` 也不會。可以同時列出兩種形式：

```bash
export CODEFYUI_EXTRA_ALLOWED_HOSTS="codefyui.example.com,codefyui.example.com:8443"
```
:::

WebSocket 交握會獨立執行相同的 `Host` 檢查，再比較瀏覽器的 `Origin` 與該 `Host`。只要代理原樣轉發 `Host`，同一個環境變數就能通過兩項檢查；因此下方 nginx 設定使用 `$http_host`，而不是 `$host`。若代理改寫 `Host`，畫布雖能載入，WebSocket 卻無法連線。拒絕原因只會顯示為 WebSocket close code：伺服器不接受 `Host` 或 `Origin` 時是 `4003`，缺少 session token 或 token 無效時是 `4401`。

## 3. 只綁定回送位址

```bash
cdui start --host 127.0.0.1 --port 8000
```

`127.0.0.1` 是預設值。代理前置後，**不要使用 `--host 0.0.0.0`**：綁定區域網路會讓任何能連到該埠的人在沒有憑證的情況下控制此實例。[發佈](./publish#6-serving-on-your-lan)說明這項取捨；使用代理就是為了移除此直接入口。

`cdui start` 會以常駐服務執行：它會脫離 terminal、寫入 pidfile，而 `cdui stop` 會先向整個行程群組送出 `SIGTERM`，必要時再改用 `SIGKILL`。它不會開啟瀏覽器，適合沒有桌面環境的伺服器。行程監督則交由下方的 systemd 處理。

## 4. 轉發 uvicorn 參數：`cdui start -- ...`

單獨一個 `--` 之後的所有參數，都會原樣轉給 uvicorn：

```bash
cdui start --host 127.0.0.1 --port 8000 -- --proxy-headers --forwarded-allow-ips 127.0.0.1
```

這能直接使用 `--proxy-headers`、`--root-path`、`--forwarded-allow-ips` 與 `--timeout-keep-alive`，不必手動啟動 uvicorn。手動啟動 uvicorn 時不會有 pidfile，也不能使用 `cdui status` 與 `cdui stop`。

這個分隔符號是雙向切開的。`cdui start` 只從 `--` **之前**的部分讀自己的參數，所以轉發過去的 `-f` 或 `--project` 絕不會被誤認成 CodefyUI 自己的參數，而將來新增的 `cdui start` 參數也絕不會蓋掉同名的 uvicorn 參數。

:::note 分隔符號之後不接受 --host 與 --port
`cdui start` 會把綁定位址記在自己的狀態檔和子行程的環境變數裡，Host 白名單也是從那裡推導出來的。如果同一個參數再從轉發那邊送一份進去，uvicorn 那邊會以後者為準，上述三處就全部對不上了，所以它會以離開碼 2 結束，並提示改用 `cdui start --host`。
:::

:::tip 使用 --proxy-headers 讓 OpenAPI 文件宣告 `https`
在 TLS 代理後方必須設定此參數。

CodefyUI 唯一一處把協定寫進回應的地方，是已發佈應用程式的 OpenAPI 文件，而它的 `servers[].url`（加上那兩段可以直接複製貼上的 `curl` 片段）是從進來那個請求的協定組出來的。uvicorn **只有**在以 `--proxy-headers` 啟動時才會用 `X-Forwarded-Proto` 改寫那個協定，而且**只**對 `--forwarded-allow-ips` 範圍內的對端這麼做。

所以：兩個參數都設了，透過 HTTPS 抓到的文件會宣告 `https://your-host/api/apps/<slug>`，Swagger UI 的「Try it out」也能用。沒設的話，即使是透過 HTTPS 連進來的，它宣告的仍然是 `http://` -- 瀏覽器接著會把那個呼叫當成混合內容擋掉，而產生出來的客戶端會拿到錯的基底網址。

CodefyUI 刻意不直接讀取 `X-Forwarded-Proto`。否則任何客戶端都能偽造此標頭，改變已發佈應用程式向文件使用者宣告的網址。由 uvicorn 處理可讓信任判斷依 `--forwarded-allow-ips` 的設定執行。

**如果代理不在這台機器上，請留意 `--forwarded-allow-ips`。** 預設值是 `127.0.0.1`，因此其他容器或主機上的代理不受信任，其 `X-Forwarded-Proto` 會被忽略，文件會改為宣告 `http://`，而且不會顯示錯誤。請將它設為代理的位址。下方 `nginx` 範例在同一台主機上終止 TLS，因此使用 `127.0.0.1`。
:::

:::note WebSocket 訊息大小
`cdui start` 與 `cdui dev` 會把 `--ws-max-size` 傳給 uvicorn，值由 `CODEFYUI_WS_MAX_MESSAGE_BYTES` 推導而來（預設：`CODEFYUI_MAX_RUN_BODY_BYTES` 是多少就是多少，也就是 64 MB）。這是畫布可以透過 `/ws/execution` 送出的最大 graph；uvicorn 自己的預設值是 16 MB，比 HTTP 本文上限*更嚴格*，會拒絕 REST API 接受的 graph。

因為這個參數是由 `cdui` 推導出來的，`--ws-max-size` 在 `--` 之後會被拒絕 -- 請改設環境變數。如果你自己手動啟動 uvicorn，除非自己傳這個參數，否則拿到的會是它的 16 MB。

如果你自行啟動 uvicorn，也請把 `CODEFYUI_HOST` / `CODEFYUI_PORT` 匯出為真正的綁定值。伺服器不會檢查自己的 socket：Host 白名單，以及只允許 loopback 的套件包與 plugin 安裝關卡，讀的都是這兩個值。執行 `uvicorn --host 0.0.0.0` 卻沒有設定 `CODEFYUI_HOST=0.0.0.0`，會讓區網 client 收到 `421`；若改用 `CODEFYUI_EXTRA_ALLOWED_HOSTS` 把它們加入白名單，安裝關卡反而會對區網保持開放，因為它們仍以為綁定的是 loopback。重新啟動模式的套件包安裝只會在 `cdui start` 下提供。

代理本身也有限制：nginx 的 `client_max_body_size` 會限制 HTTP 本文大小，而大型 WebSocket 訊框需要足夠的 `proxy_read_timeout` 才能傳完。
:::

## systemd 單元

已測試：在真的 systemd 上安裝、`systemctl enable`、啟動與停止都跑過，`systemd-analyze verify` 也沒有問題。

```ini
[Unit]
Description=CodefyUI
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
User=codefyui
Group=codefyui
WorkingDirectory=/opt/codefyui

# 對外的主機名稱，格式就是瀏覽器會放進 Host 的那一份。
# 沒設這個的話每個請求都會拿到 421 -- 包含網頁本身 --
# 瀏覽器上就是一片空白。
Environment=CODEFYUI_EXTRA_ALLOWED_HOSTS=codefyui.example.com

# 任何機密都該放這裡，而不是上面的 Environment=：那個誰都能用
# `systemctl show` 讀到。開頭的 '-' 表示檔案不存在也沒關係。
# 建議權限 0640，擁有者 root:codefyui。
EnvironmentFile=-/etc/codefyui/codefyui.env

# 刻意用 --foreground：看著這個行程的是 systemd，所以 cdui 不能自己
# 跑到背景去。只綁定回送位址 -- nginx 是唯一的入口。
ExecStart=/opt/codefyui/cdui start --foreground --host 127.0.0.1 --port 8000 -- --proxy-headers --forwarded-allow-ips 127.0.0.1

Restart=on-failure
RestartSec=5s
# 收尾一個正在跑的 run 需要的時間比預設的 90 秒長。
TimeoutStopSec=120s

# 保守的加固。刻意不加 PrivateDevices（會擋掉 GPU）、
# ProtectHome 與 ProtectSystem=strict（兩者都會擋掉對安裝目錄的寫入；
# 對這個 unit 而言，使用者資料目錄 /opt/codefyui/.codefyui_dev/ 也在其中）。
NoNewPrivileges=yes
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
```

```bash
sudo install -m 644 codefyui.service /etc/systemd/system/codefyui.service
sudo systemctl daemon-reload
sudo systemctl enable --now codefyui
systemctl status codefyui
journalctl -u codefyui -f
```

`--foreground` 是必要參數。`cdui start` 預設會以常駐模式 double-fork，不符合 `Type=exec` 服務的需求。CodefyUI 的 stdout 會寫入 journal，因此啟動錯誤與實際生效的 Host 白名單可在 `journalctl -u codefyui` 中查看。每個請求的紀錄不會出現在這裡，原因見後面的「代理同時也是你的存取紀錄」一節。

## nginx 站台設定

已測試：`nginx -t` 通過，而且底下每一個請求都是真的透過它、以 TLS 連到一台執行中的 CodefyUI。

```nginx
# /etc/nginx/sites-available/codefyui（再連結到 sites-enabled/）

# 這兩段都屬於 http{} 範圍，而 Debian/Ubuntu 的 sites-enabled/ include
# 本來就已經在那個範圍裡。

# 把 Upgrade 標頭對應到 Connection，給 WebSocket 交握用。
map $http_upgrade $connection_upgrade {
    default upgrade;
    ''      close;
}

# 跟 `combined` 一樣，但記錄的是 $uri（只有路徑）而不是 $request
# （含查詢字串）。WebSocket 的網址會用 ?token=... 帶著工作階段權杖，
# 用預設格式的話這個憑證就會被寫進一個通常會被送去集中式紀錄系統
# 的檔案裡。
log_format codefyui_noquery
    '$remote_addr - $remote_user [$time_local] '
    '"$request_method $uri $server_protocol" '
    '$status $body_bytes_sent "$http_referer" "$http_user_agent"';

server {
    listen 80;
    listen [::]:80;
    server_name codefyui.example.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    listen [::]:443 ssl;
    http2 on;
    server_name codefyui.example.com;

    ssl_certificate     /etc/letsencrypt/live/codefyui.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/codefyui.example.com/privkey.pem;

    # 資料集與模型檢查點都會經過這裡。nginx 的 client_max_body_size 預設是
    # 1 MB，超過就直接回一個沒有說明的 413。
    client_max_body_size 2g;

    # 這就是你的存取紀錄。CodefyUI 自己不會寫。
    access_log /var/log/nginx/codefyui.access.log codefyui_noquery;
    error_log  /var/log/nginx/codefyui.error.log;

    # 用 $http_host，不要用 $host：它會逐位元組轉發 Host 標頭，含非預設埠
    # 號。後端是拿 Host 去做完整字串比對，WebSocket 交握又會拿 Origin 去跟
    # 同一個 Host 比。$host 會把埠號拿掉，所以只要監聽埠不是 443，這兩關就
    # 都會壞掉。
    proxy_set_header Host              $http_host;
    proxy_set_header X-Real-IP         $remote_addr;
    proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    # /ws/execution 負責傳送每次執行的進度。少了這三行，畫布載得起來，然後
    # 什麼進度都不會回報。
    proxy_http_version 1.1;
    proxy_set_header Upgrade    $http_upgrade;
    proxy_set_header Connection $connection_upgrade;

    # 一次訓練會把連線一路開著、邊跑邊送。預設的 60 秒會在中途把它切斷。
    proxy_read_timeout 24h;
    proxy_send_timeout 24h;
    proxy_buffering    off;

    location / {
        proxy_pass http://127.0.0.1:8000;
    }
}
```

請依組織既有方式，在 `location /` 前加入單一登入，例如對 OIDC 輔助服務使用 `auth_request`、vouch 類型的 forward-auth，或使用供應商的 nginx 模組。CodefyUI 沒有身分模型，不需要感知或整合這一層。

## TLS 在代理這一層結束

為了 HTTPS，前端不需要重新 build，也不需要改任何設定。WebSocket 的網址是在執行時從 `window.location` 推導出來的，所以 `wss://` 會自動跟著 `https://` 走，埠號也會一起帶過去。隨附的 bundle 裡沒有任何寫死的主機或協定 -- 整個專案裡唯一的 `ws://localhost:8000` 屬於 Vite 開發代理，那個永遠不會被打包出去。

CodefyUI 自己不處理 TLS，也沒有憑證相關的選項。到回送位址那一段請維持純 HTTP。

## 代理同時也是你的存取紀錄

**CodefyUI 不會寫 HTTP 存取紀錄。** uvicorn 的 `uvicorn.access` logger 在啟動時被提高到 `WARNING`，而存取紀錄是以 `INFO` 輸出的，所以每一筆請求的紀錄都被丟掉了。後端裡也沒有任何會記錄請求的 middleware。這件事有實測過：透過代理送出一批請求，nginx 每一個請求都留下一行紀錄，CodefyUI 一行都沒有。

因此，上方 nginx 設定中的 `access_log` 必須保留。它是唯一的逐請求紀錄；沒有它，共用伺服器就沒有稽核軌跡。

:::warning 不要把查詢字串記下來
WebSocket 的網址會用 `?token=...` 帶著工作階段權杖。用 nginx 預設的 `combined` 格式，這個憑證會以明文寫進存取紀錄，而光是工作階段權杖就足以接管整台伺服器。上面的 `codefyui_noquery` 格式改記 `$uri` 而不是 `$request`，就是為了避免這件事。如果你用的是別的代理，請在那邊做等價的設定。
:::

CodefyUI *會*記錄的東西 -- 啟動過程、實際生效的 Host 白名單、被拒絕的 `Host` 值、警告與錯誤 -- 都輸出到標準錯誤，在 systemd 底下也就是 journal。`CODEFYUI_LOG_LEVEL`（`DEBUG` / `INFO` / `WARNING` / `ERROR`，預設 `INFO`）設定應用程式 logger 的層級 -- uvicorn 本身的詳細程度則用 `cdui start -- --log-level ...` -- 而 `CODEFYUI_LOG_JSON=1` 會改成每行一個 JSON 物件（`timestamp`、`level`、`name`、`message`、`exception`）。設定 `CODEFYUI_LOG_DIR` 會再產生一份可輪替的檔案 `<dir>/codefyui.log`（10 MB，保留五份）。未使用 `--foreground` 時，伺服器印出的所有內容都會寫入 `<install dir>/.codefyui_dev/server.log` -- `cdui start` 與 `cdui status` 都會印出這個路徑。

## 身分驗證是代理的工作，而它有極限

代理可以決定**誰能連入** CodefyUI，但無法讓 CodefyUI 對不同的已驗證使用者套用不同權限，因為 CodefyUI 沒有使用者模型。

具體來說，一旦有人通過了你們的單一登入：

- 所有存下來的圖、模型、資料集與執行紀錄，每個人都看得到也改得動。
- 環境層級的憑證是整台伺服器共用的。ChatGPT 登入、環境變數裡的 `OPENAI_API_KEY` 或 `ANTHROPIC_API_KEY`、以及 Kaggle 憑證，都屬於這台伺服器而不是某個人 -- 一個人登入之後，所有人的圖都算在他頭上，而且沒有任何紀錄說得出錢是誰花的。
- 套件包安裝的紀錄是公開的讀取。`GET /api/packs/jobs/{id}/events` 不需要工作階段權杖，和 `GET /api/runs/{id}/events` 完全一樣 -- 兩者都是讀取，套件中心就是靠輪詢它們畫出進度條。**發動**安裝仍然有把關（要工作階段權杖，而且必須綁定回送位址，除非用 `CODEFYUI_ALLOW_REMOTE_PACK_INSTALL=1` 明確放行），但安裝留下的那份紀錄會寫出它是對哪個直譯器執行的（`uv` 參數裡的 venv 路徑），也會原封不動帶著 `uv` 自己的輸出。綁定到區域網路時，任何連得到那個埠的人都讀得到它。

[共用的伺服器](./shared-instances)詳細說明這些憑證，包括讀取順序與儲存位置。將網址提供給團隊前請先閱讀。如果需要個別使用者歸屬，請為每個人執行獨立實例，使用不同的環境變數檔、`CODEFYUI_USER_DATA_DIR` 與埠號，並在每個實例前配置代理。

## CodefyUI 會往外送什麼、送給誰

共用伺服器的審查也應確認下列對外流量：

**沒有遙測、沒有分析、沒有回報主機、也沒有啟動時的更新檢查。** 伺服器在啟動時和背景都不會發出任何對外請求，瀏覽器端除了自己被載入的來源以外不會連任何其他來源。兩邊的依賴清單裡都沒有任何分析用的套件。

只有在有人主動要求時，流量才會離開這台機器：

| 什麼 | 什麼時候 |
| --- | --- |
| LLM 供應商（OpenAI、Anthropic、OpenRouter、ChatGPT，或你自己填的網址） | 執行 `LLMChat` 節點，或在設定裡列出模型時。沒有設金鑰或登入就不會發生。 |
| 資料集與模型下載（Kaggle、Hugging Face、torchvision） | 執行含有這些節點的圖時。 |
| `github.com` | `cdui install`、`cdui update`、`cdui plugin install` / `info` / `update`、`cdui project restore`，以及外掛中心。外掛抓取未經驗證時，GitHub 每個 IP 每小時只允許 60 次 API 請求，同一個 NAT 後面的所有機器共用這個額度；請在執行 `cdui start` 前匯出 `CODEFYUI_GITHUB_TOKEN`（只要有讀取公開 repo 的權限即可）。token 的處理方式見 [GitHub API 速率限制](/advanced/plugins#how-an-install-runs)。 |
| `astral.sh` | 只有在 `PATH` 上找不到 `uv` 時才會發生，正常安裝不會。那是一次性的工具鏈下載，不是回報。 |

因此，隔離網路的安裝只需避免使用上述節點與指令，不需要停用額外的回報管道。

## 檢查清單

- [ ] 一個指向代理的 DNS 名稱，而不是子路徑。
- [ ] `CODEFYUI_EXTRA_ALLOWED_HOSTS` 設成那個名稱；代理若不是監聽 443，要連埠號一起寫。
- [ ] `cdui start --host 127.0.0.1`，絕對不要 `0.0.0.0`。
- [ ] 由 systemd 管理行程，並且帶 `--foreground`。
- [ ] `proxy_set_header Host $http_host` 以及那三行 WebSocket 設定。
- [ ] `client_max_body_size` 調到 1 MB 以上。
- [ ] 代理的讀取逾時比你最長的一次訓練還久。
- [ ] 存取紀錄要開，查詢字串要關。
- [ ] 所有通過單一登入的人都已經被告知他們共用同一個身分 -- [共用的伺服器](./shared-instances)。
