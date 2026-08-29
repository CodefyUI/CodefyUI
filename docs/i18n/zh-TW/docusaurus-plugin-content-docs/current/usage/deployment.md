---
sidebar_position: 7.65
title: 放在反向代理後面
description: 給 CodefyUI 一個自己的網域名稱、只綁定回送位址，再用 nginx 在前面處理 TLS 與單一登入 -- 附上實際測試過的 systemd 單元與 nginx 設定。
---

# 放在反向代理後面

CodefyUI 沒有使用者帳號。沒有登入、沒有「這是誰的東西」，後端裡也沒有任何東西答得出「這次是誰執行的」-- 這在實務上代表什麼，請見[共用的伺服器](./shared-instances)。所以當一個團隊想在共用的伺服器前面加上真正的身分驗證，答案不是 CodefyUI 的某個功能，而是一台已經會講你們身分系統的反向代理，然後把 CodefyUI 綁在它後面的回送位址上。

這一頁就是在講怎麼做。底下的東西都在真的 nginx 和真的 systemd 上測過；有但書的地方都會明講。

```text
                       TLS + 你們的單一登入            純 HTTP，只走回送位址
  瀏覽器  ────────────────────────────────►  nginx  ──────────────────────────►  cdui start
           https://codefyui.example.com        :443                              127.0.0.1:8000
```

有三件事必須成立，而第二件是幾乎每個人都會踩的。

1. CodefyUI 要有自己的**網域名稱**，不能掛在子路徑底下。
2. 那個名稱必須加進 CodefyUI 的 **Host 白名單**，否則每一個請求都會失敗。
3. CodefyUI 只綁定**回送位址**，讓代理是唯一的入口。

## 1. 給它一個網域名稱，不要用子路徑

`https://codefyui.example.com/` 可以。`https://tools.example.com/codefyui/` 不行。

build 出來的前端產生的是根目錄絕對路徑：`index.html` 連的是 `/assets/index-*.js`，後端也是掛在字面上的 `/assets`，API 客戶端的基底路徑則是字面上的 `/api`。這裡沒有 `base` 設定，也沒有任何 build 期的環境變數可以改，所以要掛在子路徑就得自己 fork 一份前端重新 build。

:::note 為什麼這件事沒有排進計畫
對這種類型的應用程式來說，「佔用一個 vhost」本來就是正常的部署形狀，成本只是一筆 DNS 紀錄。要支援子路徑，等於要在前端、API 客戶端和後端的靜態掛載之間拉一條 build 期的設定，而多數團隊並不需要那種部署。如果你真的沒辦法配一個網域名稱，請開一個 issue 說明原因 -- 那才是會改變這個決定的輸入。
:::

## 2. 把對外的主機名稱加進白名單

**就是這一步會把一個裝好的環境變成一片空白的網頁。**

CodefyUI 會檢查每個請求的 `Host` 標頭，凡是不認得的一律回 `421 Misdirected Request`：

```json
{"detail": "Misdirected Request (Host not allowed)"}
```

這個檢查是刻意的 -- 它擋的是 DNS rebinding -- 但它跑在最外層，所以連**網頁本身**都算在內，不是只有 API。放在代理後面時 `Host` 會是你對外的名稱，而它不在預設白名單裡，於是瀏覽器拿到的是一段 JSON 而不是 `index.html`，畫面上什麼都不會出現。介面上不會有錯誤訊息，因為根本沒有介面。

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

:::warning 埠號這條規則最容易在測試時咬人
在 `:443` 或 `:80` 上，瀏覽器送的是不帶埠號的 `Host: codefyui.example.com`，所以白名單要寫沒有埠號的名稱。在其他埠上它送的是 `Host: codefyui.example.com:8443`，你就必須把**帶埠號的那個**寫進白名單。只寫 `codefyui.example.com` 不會相符，寫 `codefyui.example.com:443` 也不會。兩種都寫進去沒有壞處，而且可以省下你一個下午：

```bash
export CODEFYUI_EXTRA_ALLOWED_HOSTS="codefyui.example.com,codefyui.example.com:8443"
```
:::

WebSocket 的交握會自己再做一次同樣的 `Host` 檢查，接著再把瀏覽器送來的 `Origin` 拿去跟同一個 `Host` 比對。這兩關都靠上面那一個變數解決 -- **前提是代理原封不動地轉發 `Host`**，這也是下面 nginx 設定用 `$http_host` 而不是 `$host` 的原因。如果你在代理層改寫了 `Host`，畫布會載入，然後安安靜靜地永遠連不上。

## 3. 只綁定回送位址

```bash
cdui start --host 127.0.0.1 --port 8000
```

`127.0.0.1` 本來就是預設值，所以真正重要的是那句否定句：前面一旦有代理，**就不要用 `--host 0.0.0.0`**。綁定區域網路等於把整台伺服器的控制權免憑證交給所有連得上這個埠的人，這個取捨在[發佈](./publish#6-serving-on-your-lan)裡有說明，而那正是加代理要拿掉的東西。

`cdui start` 是真正的常駐服務：它會脫離 terminal、寫 pidfile，而 `cdui stop` 會對整個行程群組先送 `SIGTERM`、必要時再升級成 `SIGKILL`。它從來不會開啟瀏覽器，所以在沒有桌面的伺服器上是安全的。它缺的是「有人看著它」-- 那是下面 systemd 的工作。

## 4. 轉發 uvicorn 參數：`cdui start -- ...`

單獨一個 `--` 之後的所有參數，都會原樣轉給 uvicorn：

```bash
cdui start --host 127.0.0.1 --port 8000 -- --proxy-headers --forwarded-allow-ips 127.0.0.1
```

這就是讓 `--proxy-headers`、`--root-path`、`--forwarded-allow-ips` 和 `--timeout-keep-alive` 變得可用的方法，而且不必自己去叫 uvicorn -- 自己叫的代價是失去 pidfile、`cdui status` 和 `cdui stop`。

這個分隔符號是雙向切開的。`cdui start` 只從 `--` **之前**的部分讀自己的參數，所以轉發過去的 `-f` 或 `--project` 絕不會被誤認成 CodefyUI 自己的參數，而將來新增的 `cdui start` 參數也絕不會蓋掉同名的 uvicorn 參數。

:::note 分隔符號之後不接受 --host 與 --port
`cdui start` 會把綁定位址記在自己的狀態檔和子行程的環境變數裡，Host 白名單也是從那裡推導出來的。如果同一個參數再從轉發那邊送一份進去，uvicorn 那邊會以後者為準，上述三處就全部對不上了，所以它會以離開碼 2 結束，並叫你改用 `cdui start --host`。
:::

:::tip --proxy-headers 現在真的有差嗎？
還是要設 -- 它是正確的做法，也是讓 `X-Forwarded-*` 在 uvicorn 這一層從「只是存在」變成「可以信任」的東西。但要知道 CodefyUI 自己的程式目前並不讀 `X-Forwarded-Proto`。唯一一處把協定寫死進回應的地方，是已發佈應用程式的 OpenAPI 文件：即使是透過 HTTPS 連進來的，它仍然會寫 `http://your-host/api/apps/<slug>`。如果你要把那份文件餵給客戶端產生器，請手動改掉那個網址。
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

# 對外的主機名稱，格式就是瀏覽器會放進 Host 的那一份。沒設這個的話每個請求
# 都會拿到 421 -- 包含網頁本身 -- 瀏覽器上就是一片空白。
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

# 保守的加固。刻意不加 PrivateDevices（會擋掉 GPU）、ProtectHome 與
# ProtectSystem=strict（兩者都會擋掉對安裝目錄和使用者資料目錄的寫入）。
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

`--foreground` 是關鍵的那個參數。`cdui start` 預設的常駐模式會 double-fork，而那正是 `Type=exec` 的服務最不希望發生的事。另外，CodefyUI 自己的標準輸出會進 journal，所以啟動時的錯誤和實際生效的 Host 白名單都可以用 `journalctl -u codefyui` 看到 -- 但每個請求的紀錄不會在那裡，原因在再下一節。

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

# 跟 `combined` 一樣，但記錄的是 $uri（只有路徑）而不是 $request（含查詢
# 字串）。WebSocket 的網址會用 ?token=... 帶著工作階段權杖，用預設格式的話
# 這個憑證就會被寫進一個通常會被送去集中式紀錄系統的檔案裡。
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

單一登入請用你們組織既有的做法加在 `location /` 前面 -- 對 OIDC 輔助服務做 `auth_request`、vouch 那類的 forward-auth，或是你們廠商的 nginx 模組都可以。CodefyUI 不需要知道它存在；它本來就沒有可以整合的身分模型，而這正是重點。

## TLS 在代理這一層結束

為了 HTTPS，前端不需要重新 build，也不需要改任何設定。WebSocket 的網址是在執行時從 `window.location` 推導出來的，所以 `wss://` 會自動跟著 `https://` 走，埠號也會一起帶過去。出貨的 bundle 裡沒有任何寫死的主機或協定 -- 整個專案裡唯一的 `ws://localhost:8000` 屬於 Vite 開發代理，那個永遠不會被打包出去。

CodefyUI 自己不處理 TLS，也沒有憑證相關的選項。到回送位址那一段請維持純 HTTP。

## 代理同時也是你的存取紀錄

**CodefyUI 不會寫 HTTP 存取紀錄。** uvicorn 的 `uvicorn.access` logger 在啟動時被提高到 `WARNING`，而存取紀錄是以 `INFO` 輸出的，所以每一筆請求的紀錄都被丟掉了。後端裡也沒有任何會記錄請求的 middleware。這件事有實測過：透過代理送出一批請求，nginx 每一個請求都留下一行紀錄，CodefyUI 一行都沒有。

所以上面 nginx 設定裡的 `access_log` 不是裝飾品。它是「某個請求曾經發生過」的唯一證據，沒有它的話，一台共用的伺服器等於完全沒有稽核軌跡。

:::warning 不要把查詢字串記下來
WebSocket 的網址會用 `?token=...` 帶著工作階段權杖。用 nginx 預設的 `combined` 格式，這個憑證會以明文寫進存取紀錄，而光是工作階段權杖就足以接管整台伺服器。上面的 `codefyui_noquery` 格式改記 `$uri` 而不是 `$request`，就是為了避免這件事。如果你用的是別的代理，請在那邊做等價的設定。
:::

CodefyUI *會*記錄的東西 -- 啟動過程、實際生效的 Host 白名單、被拒絕的 `Host` 值、警告與錯誤 -- 都輸出到標準錯誤，在 systemd 底下也就是 journal。設定 `CODEFYUI_LOG_DIR` 可以再多一份會輪替的檔案（10 MB，保留五份）。

## 身分驗證是代理的工作，而它有極限

代理能決定的是**誰進得來**。它沒辦法讓 CodefyUI 用不同方式對待兩個已經通過驗證的人，因為這裡沒有使用者的概念可以掛。

具體來說，一旦有人通過了你們的單一登入：

- 所有存下來的圖、模型、資料集與執行紀錄，每個人都看得到也改得動。
- 環境層級的憑證是整台伺服器共用的。ChatGPT 登入、環境變數裡的 `OPENAI_API_KEY` 或 `ANTHROPIC_API_KEY`、以及 Kaggle 憑證，都屬於這台伺服器而不是某個人 -- 一個人登入之後，所有人的圖都算在他頭上，而且沒有任何紀錄說得出錢是誰花的。
- 套件包安裝的紀錄是公開的讀取。`GET /api/packs/jobs/{id}/events` 不需要工作階段權杖，和 `GET /api/runs/{id}/events` 完全一樣 -- 兩者都是讀取，套件中心就是靠輪詢它們畫出進度條。**發動**安裝仍然有把關（要工作階段權杖，而且必須綁定回送位址，除非用 `CODEFYUI_ALLOW_REMOTE_PACK_INSTALL=1` 明確放行），但安裝留下的那份紀錄會寫出它是對哪個直譯器執行的（`uv` 參數裡的 venv 路徑），也會原封不動帶著 `uv` 自己的輸出。綁定到區域網路時，任何連得到那個埠的人都讀得到它。

[共用的伺服器](./shared-instances)把這些憑證講得更細，包含讀取的先後順序和各自存在哪裡。把網址發給一個團隊之前請先讀那一頁。它對「我們需要分辨是誰用的」給的答案在這裡同樣成立：一個人一台伺服器，各自獨立的環境變數檔、各自獨立的 `CODEFYUI_USER_DATA_DIR`、各自獨立的埠號 -- 然後每一台前面各放一台代理。

## CodefyUI 會往外送什麼、送給誰

值得明講，因為這通常會跟「我們能不能把它放到共用伺服器上」在同一次審查裡被問到：

**沒有遙測、沒有分析、沒有回報主機、也沒有啟動時的更新檢查。** 伺服器在啟動時和背景都不會發出任何對外請求，瀏覽器端除了自己被載入的來源以外不會連任何其他來源。兩邊的依賴清單裡都沒有任何分析用的套件。

只有在有人主動要求時，流量才會離開這台機器：

| 什麼 | 什麼時候 |
| --- | --- |
| LLM 供應商（OpenAI、Anthropic、OpenRouter、ChatGPT，或你自己填的網址） | 執行 `LLMChat` 節點，或在設定裡列出模型時。沒有設金鑰或登入就不會發生。 |
| 資料集與模型下載（Kaggle、Hugging Face、torchvision） | 執行含有這些節點的圖時。 |
| `github.com` | `cdui install`、`cdui update`、`cdui plugin install`。 |
| `astral.sh` | 只有在 `PATH` 上找不到 `uv` 時才會發生，正常安裝不會。那是一次性的工具鏈下載，不是回報。 |

所以要做到完全離線，是「不要用那些節點」的問題，而不是「要關掉某個回報管道」的問題。

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
