---
sidebar_position: 7.6
title: 發佈（把 graph 變成應用程式）
description: 把已儲存的 graph 固定為有版本的應用程式，透過穩定且受 API key 保護的 invoke 端點提供服務，並把每次執行記錄到 SQLite。
---

# 發佈（把 graph 變成應用程式）

[把 graph 當成函式呼叫](./graph-as-a-function)讓任何已儲存的 graph 都能透過 HTTP 呼叫，但該端點不固定：每次從畫布存檔都會改變其執行內容，而工作階段權杖（session token）會在每次伺服器重新啟動時輪換。**發佈**會把可正常執行的 graph 轉成穩定的產品端點：

- 一份不可變的**版本**快照，透過 `POST /api/apps/{slug}/invoke` 提供服務，
- 由重新啟動後仍有效的長效 **API key**（`Authorization: Bearer cdui_...`）保護，
- 每個已解析至版本的 invoke 都會記錄到 SQLite（`backend/data/codefyui.db`）。

畫布編輯只會變更已儲存的 graph 檔案；invoke 只會讀取儲存的快照。除非重新發佈，否則編輯畫布不會改變已發佈的應用程式。

以下所有管理呼叫都使用編輯器的工作階段權杖（`X-CodefyUI-Token`；取得方式與[把 graph 當成函式呼叫](./graph-as-a-function#2-getting-the-token-for-external-scripts)頁面相同）。Windows 請使用 `curl.exe`，並以檔案傳入 request body（`--data "@payload.json"`），不要使用行內 JSON。

## 1. 發佈的生命週期 {/* #1-publish-lifecycle */}

```text
POST /api/apps/{slug}/publish        (session token)
body: {"graph": "<saved name>", "note": "optional", "create": false}
      (optional "record_io": true|false -- omitted inherits the app's current setting; see below)
      (optional "git_commit": "<7-40 hex>", "git_dirty": true|false -- publish provenance;
       normally set for you by `cdui project publish`)
```

- `slug` 是穩定的對外名稱：`^[a-z][a-z0-9-]{0,63}$`。它由你選定，與 graph 名稱無關；重新命名 graph 不會使已發佈的網址失效。
- 首次發佈至不存在的 slug 時必須傳入 `"create": true`，否則會收到 404 `app_not_found`。重新發佈時，即使拼錯 slug，也不會自動建立第二個應用程式。
- 發佈到既有 slug 時會新增下一個版本；這就是重新發佈流程。
- `note` 是選填、不可變的版本中繼資料，會在版本清單裡原樣回顯。
- 發佈會先執行與 `/run` 相同的預檢：會被 `POST /api/graph/run/{name}` 拒絕的 graph（409 `invalid_contract` / `no_entry_points` / `untriggered_input` / `unreachable_output` / `invalid_graph`）也不能發佈。
- 發佈也會拒絕（409 `secret_in_graph`）一張存檔裡仍然帶有非空 `SECRET` 型別參數（例如 LLMChat 的 API key）的 graph，並指出是哪個節點的哪個參數。**graph 永遠不會儲存機密** -- 編輯器會遮罩 `SECRET` 欄位並在存檔／匯出時移除它們，`POST /api/graph/save` 也會在伺服器端移除它們，因此這項預檢只會在手動編輯過的檔案上觸發。請改用環境變數存放 API key（`CODEFYUI_OPENAI_API_KEY` / `OPENAI_API_KEY`、`CODEFYUI_ANTHROPIC_API_KEY` / `ANTHROPIC_API_KEY`）。
- 成功：`{"slug", "version", "active": true, "created", "graph_name", "note", "git_commit", "git_dirty"}`。
- **溯源。** `git_commit`（驗證格式 `^[0-9a-f]{7,40}$`；不合法的值會收到 422 `invalid_git_commit`）與 `git_dirty` 會記錄在版本上。`cdui project publish` 會從 `git rev-parse HEAD` + `git status --porcelain` 填入這兩個欄位，並在 working tree 不乾淨時明確警告。它們會出現在 `GET /api/apps/{slug}/versions`，也會以 `x-codefyui-git-commit` / `x-codefyui-git-dirty` 出現在各應用程式 OpenAPI 文件的 `info` 區塊裡。請見[專案目錄](./project-directories)。

**發佈會立即生效。** v1 沒有 staging 流程；請用畫布的**執行**加上相同的預檢完成驗證。如果 graph 能從**執行**按鈕執行，而且 `/run` 接受它，已發佈版本就會提供相同行為。

```powershell
# payload.json: {"graph": "my-classifier", "create": true, "note": "first cut"}
$token = Get-Content "$HOME\CodefyUI\.codefyui_dev\session.token"
curl.exe -s -X POST "http://127.0.0.1:8000/api/apps/classifier/publish" `
  -H "X-CodefyUI-Token: $token" -H "Content-Type: application/json" `
  --data "@payload.json"
```

```bash
curl -s -X POST "http://127.0.0.1:8000/api/apps/classifier/publish" \
  -H "X-CodefyUI-Token: $(cat ~/CodefyUI/.codefyui_dev/session.token)" \
  -H "Content-Type: application/json" \
  --data "@payload.json"
```

管理版本：

```text
GET    /api/apps                       -> [{slug, graph_name, active_version, versions_count, record_io, ...}]
GET    /api/apps/{slug}/versions       -> [{version, source_graph_name, note, git_commit, git_dirty, created_at, active}]
POST   /api/apps/{slug}/activate       body {"version": n} -- point the slug at ANY existing version
POST   /api/apps/{slug}/unpublish      -> active_version = null; versions and runs are kept
PATCH  /api/apps/{slug}                body {"record_io": bool} -- flips run-recording, no republish
DELETE /api/apps/{slug}
```

- `activate` 也用於回滾：啟用較舊的版本會還原該版本；從未發佈狀態啟用時，則會以該版本恢復服務。
- 發佈時的 `record_io` 有三種狀態：省略它（或傳 `null`）會**繼承**應用程式目前的 `record_io`。重新發佈時若未指定此欄位，就不會改變你透過 `PATCH` 設定的值。全新的應用程式（以 `"create": true` 第一次發佈）在省略時預設為 `true`。無論建立或重新發佈，都可傳入明確的 `true`／`false` 覆寫設定。
- 未發佈期間，invoke 會回 409 `app_unpublished` -- 版本與執行紀錄都會保留。
- **`DELETE /api/apps/{slug}` 會不可逆地移除這個應用程式、它所有的版本，以及所有的執行紀錄。** 沒有復原功能。除非確定要永久刪除，否則請優先使用 `unpublish`。

## 2. API key {/* #2-api-keys */}

```text
POST /api/keys                (session token)  body {"name": "ci-bot"}
GET  /api/keys                (session token)  -- id, name, prefix, timestamps; never secrets
POST /api/keys/{id}/revoke    (session token)  -- soft revoke; the row stays listed
```

**完整的 key（`cdui_...`）只會在建立回應中出現一次，而且不會儲存或寫入 log。** 請立即複製；清單只會顯示前 12 個字元（`prefix`）。key 以 sha256 雜湊儲存，重新啟動後仍有效；這點與會輪換的工作階段權杖不同。撤銷後，key 會立即無法通過驗證，但仍會帶著 `revoked_at` 留在清單中，因此舊的執行紀錄仍可追溯到該 key。

## 3. Invoke {/* #3-invoke */}

```text
POST /api/apps/{slug}/invoke          (auth: Authorization: Bearer cdui_...)
```

request body 與 [`/api/graph/run`](./graph-as-a-function) 相同：整個 body 為選填，其中 `inputs`、`timeout_s`、`device` 也都是選填欄位。兩者有以下差異：

- `record_outputs` **會被接受但忽略**；已發佈的執行會記錄在 SQLite（見下方），不會寫入編輯器的檢視器儲存區。
- `timeout_s` 涵蓋**包含排隊等待在內**的完整請求時間。同一個應用程式的 invoke 由每個 slug 各自的鎖逐一執行；若呼叫在等待前一個 invoke 時用完時間，會回傳 `timeout` 錯誤，並註明逾時發生在排隊期間。不同 slug 可以平行執行。

invoke **不接受**編輯器的工作階段權杖；使用該權杖時會收到 401，回應會直接指出原因："this endpoint takes an API key (cdui_...), not the editor session token"。

每個回應都採用與 `/run` 相同、包含 9 個鍵的封裝格式（見[回應封裝格式](./graph-as-a-function)），並填入第二階段新增的兩個欄位：`graph` 與 `app` 在每種結果中都是 slug，`version` 則是實際執行的版本（若錯誤發生在解析版本前，則為 `null`）。除了第一階段的錯誤分類外，另新增下列錯誤代碼：

| `error.code` | HTTP | 觸發條件 |
| --- | --- | --- |
| `invalid_key` | 401 | bearer token 缺失／格式錯誤／不認識／已撤銷（回應帶有 `WWW-Authenticate: Bearer`） |
| `app_not_found` | 404 | slug 不存在 |
| `app_unpublished` | 409 | 應用程式存在，但沒有任何版本是啟用中的 |

過大的圖片在**兩條**執行路由上都會一開始就被拒絕：單一張圖片超過 `MAX_IMAGE_PIXELS`（預設 25,000,000；`CODEFYUI_MAX_IMAGE_PIXELS`）時回 422 `invalid_input`。請比對代碼而不是訊息 -- 遠超過預算時，出現的會是 PIL 自己的 decompression-bomb 錯誤文字，而不是我們的。

PowerShell：

```powershell
# payload.json: {"inputs": {"x": "hello"}}
curl.exe -s -X POST "http://127.0.0.1:8000/api/apps/classifier/invoke" `
  -H "Authorization: Bearer cdui_YOUR_KEY" `
  -H "Content-Type: application/json" `
  --data "@payload.json"
```

bash：

```bash
curl -s -X POST "http://127.0.0.1:8000/api/apps/classifier/invoke" \
  -H "Authorization: Bearer cdui_YOUR_KEY" \
  -H "Content-Type: application/json" \
  --data "@payload.json"
```

## 4. 執行紀錄 {/* #4-run-records */}

每個已解析至應用程式版本的 invoke 都會寫入一列，包含狀態、錯誤代碼、裝置、`total_s`、各節點計時、受大小限制的輸入／輸出，以及 key id。解析版本前就遭拒絕的請求（`invalid_key`、`app_not_found`、`app_unpublished`）不會寫入資料。記錄採盡力而為方式；如果執行後無法儲存，系統會寫入 log，但仍會回傳執行結果。

```text
GET /api/apps/{slug}/runs?limit=50&before=<created_at>&before_id=<run_id>   -- newest-first metadata only
GET /api/apps/{slug}/runs/{run_id}                                         -- the full row incl. inputs/outputs/node_timings
```

`before` 與 `before_id` 是上一頁最後一列的 `created_at` 與 `run_id`。兩者合起來在「最新在前」的排序裡指定了唯一的一列，所以就算分頁邊界落在一群記錄於同一個時間戳刻度的執行中間，也能從正確的位置接續。只送 `before` 仍然被接受，供依照較早合約寫成的客戶端使用，而且保留原本的意義：嚴格早於那個時間戳的列，也就是會略過任何共用該時間戳的執行。只送 `before_id` 則是 422 `incomplete_cursor`。

讀取時**兩者皆可**：一把有效的 API key，或編輯器的工作階段權杖（編輯器介面讀取執行紀錄時從不持有 API key），兩者都沒有的請求會被拒絕（401，純 `{"detail": ...}`）。不認識的 slug 在兩條路由上都是 404 `app_not_found`；不認識的 `run_id` 則是 404 `run_not_found`。

儲存的輸入／輸出以每個欄位 `RUN_IO_CAP_BYTES` 為上限（預設 64 KB；base64 圖片本身已有上限）。未儲存的欄位仍會以固定的標記物件表示，並保持可解析：

- 超過上限：`{"__codefyui__": "truncated", "bytes": N}`
- 因為應用程式設了 `record_io: false` 而被隱去：`{"__codefyui__": "redacted"}`

保留期限：`CODEFYUI_RUNS_RETENTION_DAYS` 預設為 **0 = 永久保留**。設成大於 0 時，較舊的資料列會在啟動時清除，也會在寫入時至多每小時清除一次。每次清除都會在 log 中明確記錄列數。

## 5. 各應用程式的 OpenAPI 文件 {/* #5-per-app-openapi-document */}

```text
GET /api/apps/{slug}/openapi.json      (API key or session token)
```

這是一份對應**啟用中**版本的完整獨立 OpenAPI 3.1 文件，可直接匯入 Swagger UI、Postman 或 openapi-generator。文件包含從 graph 合約推導出的型別化輸入 schema、具有 9 個鍵的回應封裝 schema、bearer 安全性方案，以及 `x-codefyui-curl` 物件；該物件提供可直接貼上的 `powershell` 與 `bash` invoke 指令。端點只回傳 JSON，不會產生 HTML 頁面。若啟用中版本在發佈時包含溯源資訊，OpenAPI 的 `info` 區塊也會包含 `x-codefyui-git-commit` 與 `x-codefyui-git-dirty`。

## 6. 在區域網路上提供服務 {/* #6-serving-on-your-lan */}

```text
cdui start --host 0.0.0.0 --port 8000      # all interfaces
cdui start --host 192.168.1.20             # one concrete interface
```

Host 標頭白名單會依綁定位址自動設定：具體的區域網路 IP 會加入白名單；綁定 `0.0.0.0` 時，則會加入每個本機網路介面的 IP。可透過 `CODEFYUI_EXTRA_ALLOWED_HOSTS="mybox:8000,192.168.1.20:8000"` 加入其他名稱。啟動時會印出實際生效的白名單與可連線網址；`cdui status` 與 `cdui stop` 也會回報實際位址。`cdui dev` 依設計只綁定回送位址。

如果不使用 `cdui start`，而是手動啟動 uvicorn，還需要把 `CODEFYUI_HOST` 與 `CODEFYUI_PORT` 設成實際綁定位址。伺服器不會檢查自己的 socket；Host 白名單，以及套件包與外掛安裝的回送位址限制，都依這兩個值判定。請見[放在反向代理後面](./deployment)。

綁定區域網路會有以下暴露範圍與限制：

- 綁定區網位址會提供**完整的**編輯器，而 `GET /api/auth/bootstrap` 會把工作階段權杖交給任何 Host 在白名單內的請求。**任何能連線到該連接埠的使用者都能控制這台伺服器；只在信任的網路上使用。**
- 因此，已發佈介面上的 API key 只用於追蹤呼叫來源與妥善管理外部腳本的憑證，**不是**區域網路存取控制。
- 傳輸是純 HTTP -- v1 沒有 TLS。
- CORS 設定不會降低此暴露範圍：暴露的是同源的部分，而 `Authorization` 這個 CORS 標頭只用於讓未來跨來源的 JS 呼叫端通過預檢 -- 它不是緩解措施。
- 區域網路存取控制**不在規劃內**（issue #247，於 2026-08-30 關閉）：伺服器不會依網路來源限制編輯器。因此，共用伺服器應採用下方的反向代理設定，而不是直接綁定區域網路。

相關憑證風險請見[共用的伺服器](./shared-instances)。一台伺服器只有**一個**身分，因此 ChatGPT 登入、`.env` 裡的 LLM 金鑰與 Kaggle 憑證，會由所有能連上該連接埠的人共用，而且不會記錄費用由誰產生。

如果需要具身分驗證的共用伺服器，請不要直接綁定區域網路。請只綁定回送位址，並在前方設定反向代理；這是目前唯一能在 CodefyUI 實例前加入身分驗證與 TLS 的方式。[放在反向代理後面](./deployment)提供完整設定，包含實際測試過的 nginx 站台設定與 systemd 單元。
