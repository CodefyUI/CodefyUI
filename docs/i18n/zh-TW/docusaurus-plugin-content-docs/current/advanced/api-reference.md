---
sidebar_position: 5
title: API 參考
description: 節點、圖、run、sweep、已發佈應用程式、外掛、選用套件包、檔案、媒體與 LLM 代理的後端 route 和驗證要求，以及 request 上限、錯誤格式與執行 WebSocket 協定。
---

# API 參考

後端提供 REST API 與執行用 WebSocket。所有端點都使用與應用程式相同的來源（預設為 `http://localhost:8000`）。**驗證**欄使用[驗證](#authentication)一節定義的五種值。每個表格都會連結到相關的使用說明頁面。

## 驗證 {/* #authentication */}

| 驗證 | 意義 |
|------|---------|
| open | 不需憑證。大多數 `GET` / `HEAD` / `OPTIONS` 請求使用這項規則；下方已發佈應用程式與 API key 的列會標出例外。 |
| token | `X-CodefyUI-Token` session header。一般 middleware 會要求 `/api/` 底下的每個 `POST` / `PUT` / `PATCH` / `DELETE` 都帶上它，但 `/api/apps` 與 `/api/keys` 由各 route 依本表其中一項規則自行驗證。所有應用程式管理 route（包含 `GET /api/apps` 與 `GET /api/apps/{slug}/versions`）以及整個 `/api/keys` 都使用這項憑證。遺漏或錯誤時：`403 {"detail": "Missing or invalid X-CodefyUI-Token header"}`。token 檔案的位置，以及它為何每次重啟都會輪替：見[把 graph 當成函式呼叫，第 2 節](/usage/graph-as-a-function#2-getting-the-token-for-external-scripts)。 |
| token+loopback | token，加上伺服器必須綁定至回送位址（`127.0.0.1`、`localhost`、`::1`）——否則回傳 403——除非事先匯出 `CODEFYUI_ALLOW_REMOTE_PACK_INSTALL=1`（套件包）或 `CODEFYUI_ALLOW_REMOTE_PLUGIN_INSTALL=1`（外掛）。此閘門讀取 `CODEFYUI_HOST`，從不讀取 socket。 |
| API key | 只接受 `Authorization: Bearer cdui_...`。使用 session token 時，錯誤訊息會指出此處需要 API key。驗證失敗會在 run envelope 內回傳 401，並包含 `WWW-Authenticate: Bearer`。key 透過 `/api/keys` 建立；請參閱[發佈，第 2 節](/usage/publish#2-api-keys)。 |
| key-or-token | 接受任一種憑證。這些 metadata 讀取是 `GET` 通常開放之外的另一組例外。兩種憑證都沒有時，會回傳 `401 {"detail": "..."}`。 |

Host guard 會在其他所有檢查之前處理每個 request，包括 SPA 頁面與 WebSocket。`Host` header 不在 allowlist 內時，會回傳 `421 {"detail": "Misdirected Request (Host not allowed)"}`。allowlist 由 `CODEFYUI_HOST` 與 `CODEFYUI_PORT` 推導，並可透過 `CODEFYUI_EXTRA_ALLOWED_HOSTS` 擴充；請參閱[發佈，第 6 節](/usage/publish#6-serving-on-your-lan)與[放在反向代理後面](/usage/deployment)。token middleware 會略過 `GET /api/auth/bootstrap`，讓前端取得 token。只要 `Host` 獲准就能呼叫此端點，包括伺服器綁定至區網位址時的遠端 client。

## 健康狀態與系統 {/* #health-and-system */}

| 端點 | 方法 | 驗證 | 說明 |
|----------|--------|------|-------------|
| `/api/health` | GET | open | 健康探測，包含 `status`、`version`、`boot_id`（用來讓 client 偵測伺服器重新啟動的行程識別碼）、`nodes_loaded`、`presets_loaded`、`caches`（各記憶體內 store 目前使用的位元組與上限；請參閱[訓練記憶體](./training-memory)），以及伺服器以 `--project` 執行時的 `project`（專案目錄絕對路徑）。 |
| `/api/auth/bootstrap` | GET | open | 對任何 Host 獲准的請求回傳 `{"token": "..."}`——前端取得 session token 的方式。 |
| `/api/system/devices` | GET | open | 可用於執行圖表的運算裝置：最佳可用的 `default`，加上一份能區分 NVIDIA CUDA、AMD ROCm 與 Apple MPS 的具名 `devices` 清單。支援編輯器的裝置選擇器。 |

## 節點與預設模組 {/* #nodes-and-presets */}

| 端點 | 方法 | 驗證 | 說明 |
|----------|--------|------|-------------|
| `/api/nodes` | GET | open | 列出所有節點定義。每個節點都帶有 `requires_pack`（執行前所需的套件包 id，沒有則為 `null`），每個 SELECT 參數則帶有 `option_packs`（選項值到套件包 id 的對應）。編輯器會讓目前使用但尚未安裝的值保持可選取並顯示警告，將其他尚未安裝的選項變灰，並提供安裝入口；無論如何，run 本身仍由後端把關。 |
| `/api/nodes/{node_name}` | GET | open | 取得單一節點定義。 |
| `/api/nodes/reload` | POST | token | 重新探索每一個節點與預設模組來源：從磁碟重新 import 自訂節點與外掛；重新註冊但不重新 import 內建節點；重新掃描預設模組。回傳 `{builtin, custom, plugins, presets, total}`；與 `POST /api/plugins/reload` 完全相同。 |
| `/api/nodes/script/validate` | POST | token | 在輸入 PythonScript body 時，依 Tier-0 政策檢查它（`{"code"}`）：`{ok, error, line, defines_run, allowed_modules}`。`ok: false` 是正常的 200，不是錯誤。 |
| `/api/presets` | GET | open | 列出預設模組定義。 |
| `/api/presets/{name}` | GET | open | 取得單一預設模組定義。 |
| `/api/presets/create` | POST | token | 從請求中的完整 `nodes` 與 `edges` 建立新預設模組；編輯器會送出目前的整張畫布。 |

## 圖表 {/* #graphs */}

| 端點 | 方法 | 驗證 | 說明 |
|----------|--------|------|-------------|
| `/api/graph/validate` | POST | token | 驗證一張圖。 |
| `/api/graph/save` | POST | token | 儲存一張圖。 |
| `/api/graph/load/{name}` | GET | open | 載入一張已儲存的圖。 |
| `/api/graph/list` | GET | open | 列出已儲存的圖。 |
| `/api/graph/export` | POST | token | 匯出單檔、headless 的 Python runner。它會內嵌圖表，並需要相容的 CodefyUI 後端環境，但不需要執行中的伺服器。 |
| `/api/graph/contract/{name}` | GET | open | 從已儲存圖表推導出的函式簽名——`inputs`、`outputs`、`problems`——供腳本呼叫。參閱[把 graph 當成函式呼叫，第 3 節](/usage/graph-as-a-function#3-inspect-the-contract)。 |
| `/api/graph/run/{name}` | POST | token | 將最新儲存的檔案當成函式執行：輸入 `{inputs, timeout_s, device, record_outputs}`，每種結果都輸出九鍵 envelope。參閱[把 graph 當成函式呼叫，第 4 節](/usage/graph-as-a-function#4-run-the-graph)。 |
| `/api/examples/list` | GET | open | 列出範例圖。 |
| `/api/examples/load` | GET | open | 載入一張範例圖。 |

## Run 與 sweep {/* #runs-and-sweeps */}

| 端點 | 方法 | 驗證 | 說明 |
|----------|--------|------|-------------|
| `/api/runs` | POST | token | 將 run 送進佇列並立即回傳：`{run_id, status: "running" \| "queued"}`。 |
| `/api/runs` | GET | open | 由新到舊列出；`?status=` 可重複，`?limit=` 上限為 500，`?offset=` 指定起始列。每列都包含 `queue_position`、`active` 與 `final_metrics`；response 也包含未分頁的 `total`。 |
| `/api/runs/{run_id}` | GET | open | 回傳單一 run 與 `last_cursor`，client 會從這個位置開始輪詢事件。 |
| `/api/runs/{run_id}` | DELETE | token | 刪除已完成的 run 及其事件、metrics、artifact 列與擷取的輸出；還在佇列或執行中則回傳 409。artifact 檔案會留在磁碟上。 |
| `/api/runs/{run_id}/cancel` | POST | token | 協作式停止——`{run_id, status, cancelled}`；若 run 早已結束，`cancelled: false`（仍為 200）。 |
| `/api/runs/{run_id}/events` | GET | open | 回傳 `?cursor=` 之後的事件；`?wait=` 最多長輪詢 60 秒，`?limit=` 上限為 2000。回傳 `{run_id, status, active, events[{cursor, type, payload, ts}], cursor}`。頁面可能少於 `limit` 筆；下一個 request 應使用回傳的 `cursor`。 |
| `/api/runs/{run_id}/metrics` | GET | open | 依 `(name, step)` 排序的已記錄純量序列；`?name=` 篩選，`?format=csv` 下載（UTF-8 BOM，儲存格可防公式注入）。JSON 形式也會列出 `names`。 |
| `/api/runs/{run_id}/artifacts` | GET | open | run 記錄的檔案（checkpoint、匯出內容、影像），由舊到新；`?kind=` 篩選，未知的 kind 會得到空清單。 |
| `/api/sweeps` | POST | token | 將參數 sweep 編譯成多個變體 run，全部排入佇列——回傳 `201`、`sweep_id` 與每個變體各一個 `run_id`。 |
| `/api/sweeps/{sweep_id}` | GET | open | 排名後的比較表——最佳優先的 `variants`、`best`、`counts`，以及存在時的 `objective_warning`；`?format=csv` 可下載。 |
| `/api/sweeps/{sweep_id}/cancel` | POST | token | 取消每一個排隊中或執行中的變體：`{sweep_id, state, cancelled, already_finished, variants[]}`，依變體索引順序每個一筆。 |

**Runs API。** `POST /api/runs` 接受 `{"graph": {...}, "options": {...}, "name": "..."}`。graph 使用已儲存圖的 JSON 格式（`nodes`、`edges`，以及選用的 `presets` 與 `subgraphs`）。envelope 或選項無效時回傳 400；run service 無法使用，或 `interactive` lane 的送出數超過上限時，回傳 503。選項 key 是封閉集合：`device`、`seed`、`deterministic`、`record_outputs`、`lane`、畫布旗標 `verbose`、`graph_id`、`weights_persistent`、`backward_mode`、`auto_backward`，以及引擎錯誤政策 `error_mode`、`max_retries`。run 的 `status` 只會是 `queued`、`running`、`succeeded`、`failed`、`cancelled` 或 `interrupted`。`/events` 有兩個上限：單一 payload 超過 `CODEFYUI_RUN_EVENT_PAYLOAD_CAP_BYTES`（128 KB）時，輸出會以省略標記取代後再儲存；response 超過 `CODEFYUI_RUN_EVENTS_RESPONSE_CAP_BYTES`（4 MB）時會結束。佇列順序、lane、保留政策與 `cdui run` 請見[執行佇列](/usage/run-queue)。

**Sweeps。** `POST /api/sweeps` 接受 `base_graph`、一份 `sweep_spec`（`method` 為 `grid` 或 `random`、`seed`、`samples`，以及 `params[{node_id, param, values | range}]`）、必要的 `objective`（`metric`，以及 `direction` 為 `minimize` 或 `maximize`）、同一組 `options`、`name` 與 `seed_variants`。最多建立 `CODEFYUI_MAX_SWEEP_RUNS`（32）個變體。每個變體都是一般的 `/api/runs` 列，可分別追蹤其 `/events` 端點。spec、驗證錯誤與取消行為請見[執行佇列——Sweeps](/usage/run-queue#sweeps)。

## 執行輸出與狀態 {/* #execution-outputs-and-state */}

| 端點 | 方法 | 驗證 | 說明 |
|----------|--------|------|-------------|
| `/api/execution/outputs/{run_id}` | GET | open | 列出某次 run 擷取的連接埠。 |
| `/api/execution/outputs/{run_id}` | DELETE | token | 清除某次擷取的 run。 |
| `/api/execution/outputs/{run_id}/{node_id}/{port}` | GET | open | 取得一個已擷取的張量（支援 `?slice=0,:,:` 與 `?max_elements=`，預設 4096、上限 1,000,000）；切片仍然過大時回傳 413。 |
| `/api/execution/outputs/{run_id}/{node_id}/{port}/stats` | GET | open | 回傳單一已擷取 port 的伺服器端摘要統計：固定的一組純量與 64-bin histogram；label tensor 則回傳值計數。無論 tensor 大小，response 通常只有一到兩 KB。超過 `CODEFYUI_STATS_SAMPLE_THRESHOLD`（4,000,000 個元素）的 tensor 會進行取樣。 |
| `/api/execution/outputs/{run_id}/{node_id}/__steps_index` | GET | open | 某節點的步驟追蹤 metadata（檢視器 → 步驟分頁）。 |
| `/api/execution/outputs/{run_id}/{node_id}/__grad_index` | GET | open | 已擷取的梯度 metadata（檢視器 → 反向分頁）。 |
| `/api/execution/state/reset` | POST | token | 重設已保存的層權重（單一節點或單一圖表）。 |
| `/api/execution/state/list` | GET | open | 列出保存了多少模組（診斷用）。 |

## 已發佈的應用程式與 API key {/* #published-apps-and-api-keys */}

| 端點 | 方法 | 驗證 | 說明 |
|----------|--------|------|-------------|
| `/api/apps` | GET | token | 列出應用程式——`slug`、`graph_name`、`active_version`、`versions_count`、`record_io`。 |
| `/api/apps/{slug}/publish` | POST | token | 將指定的已儲存圖表凍結成該 slug 的下一版並啟用（新 slug 使用 `"create": true`）；先執行 `/run` 的前置檢查。 |
| `/api/apps/{slug}/versions` | GET | token | 每一個版本及其 note、provenance（`git_commit`、`git_dirty`）與 `active` 旗標。 |
| `/api/apps/{slug}/activate` | POST | token | 將 slug 指向任一既有版本（`{"version": n}`）——也是 rollback 路徑。 |
| `/api/apps/{slug}/unpublish` | POST | token | 設定 `active_version = null`；版本與 run 都會保留。 |
| `/api/apps/{slug}` | PATCH | token | 不需重新發佈即可切換 `record_io`。 |
| `/api/apps/{slug}` | DELETE | token | 移除應用程式、它的所有版本與所有 run 紀錄——無法復原。 |
| `/api/apps/{slug}/invoke` | POST | API key | 執行作用中的版本：body 與 envelope 都和 `/api/graph/run/{name}` 相同，並填入 `app` 與 `version`。 |
| `/api/apps/{slug}/openapi.json` | GET | key-or-token | 作用中版本的一份獨立 OpenAPI 3.1 文件。 |
| `/api/apps/{slug}/runs` | GET | key-or-token | 由新到舊的 run 紀錄，僅含 metadata；以 `?before=` 與 `?before_id=` 分頁。 |
| `/api/apps/{slug}/runs/{run_id}` | GET | key-or-token | 單一紀錄，包含其輸入、輸出與節點 timings。 |
| `/api/keys` | POST | token | 產生 key（`{"name"}`）；完整的 `cdui_...` token 只會在這次回應中出現。 |
| `/api/keys` | GET | token | 列出 key——`id`、`name`、`prefix`、timestamps，絕不含 secret；已撤銷的列仍會列出。 |
| `/api/keys/{key_id}/revoke` | POST | token | 軟性撤銷；找不到時回傳 404 `key_not_found`。 |

生命週期、run 紀錄、分頁與每個應用程式的 OpenAPI 文件，請見[發佈](/usage/publish)。

## 自訂節點與外掛 {/* #custom-nodes-and-plugins */}

| 端點 | 方法 | 驗證 | 說明 |
|----------|--------|------|-------------|
| `/api/custom-nodes` | GET | open | 列出自訂節點。 |
| `/api/custom-nodes/upload` | POST | token | 上傳一個自訂節點。 |
| `/api/custom-nodes/toggle` | POST | token | 啟用／停用一個自訂節點。 |
| `/api/custom-nodes/{filename}` | DELETE | token | 刪除一個自訂節點。 |
| `/api/plugins` | GET | open | 列出已安裝的外掛包。 |
| `/api/plugins/catalog` | GET | open | 列出型錄與已安裝的外掛。每列包含其狀態：已安裝、已停用、明確移除或檔案遺失。 |
| `/api/plugins/generation` | GET | open | 回傳編輯器用來輪詢節點面板變更的 reload generation。 |
| `/api/plugins/{id}` | GET | open | 取得某外掛的 manifest 與 README。 |
| `/api/plugins/reload` | POST | token | 與 `POST /api/nodes/reload` 相同。 |
| `/api/plugins/inspect` | POST | token+loopback | 在單一個已解析 commit 上檢查型錄名稱、`owner/repo` 或 URL。回傳安裝需求並存入 `inspection_id`，但不安裝外掛。 |
| `/api/plugins/install` | POST | token+loopback | 安裝 `inspection_id` 指定的 manifest；回傳 `202` 與 `job_id`。伺服器使用已檢查的 manifest，不接受這個 request 提供安裝 metadata。 |
| `/api/plugins/jobs/{job_id}/events` | GET | open | 回傳安裝 job 在 `?cursor=` 之後的 log 與進度。`?wait=` 最多長輪詢 60 秒。job 可能以 `needs_restart` 結束，並包含伺服器停止後要執行的指令。 |
| `/api/plugins/jobs/{job_id}/cancel` | POST | token+loopback | 取消執行中的安裝並移除部分寫入的內容。 |
| `/api/plugins/{id}/update` | POST | token+loopback | 檢查外掛的 GitHub repository 是否有更新。回傳 `202 {"job_id"}`、`200 {"status": "up_to_date", "sha"}`，或在更新需要額外權限時回傳 `200 {"status": "needs_consent", "inspection", "capabilities_added", "allowed_modules_added"}`。若需同意，請呼叫 `POST /api/plugins/install {"inspection_id", "accept_capabilities", "trust_author"}` 完成更新，不需使用 `force`。更新會保留啟用狀態。內建外掛、本機連結外掛，或 manifest 已改用另一個外掛 id 的 repository，會回傳 `400 not_updatable`。 |
| `/api/plugins/{id}` | DELETE | token+loopback | 解除安裝外掛。內建外掛會保留檔案並記錄為已移除；本機連結目錄不會變更。Python 套件也會保留，response 會提供解除安裝這些套件的指令。 |
| `/api/plugins/{id}/enable` | POST | token | 啟用一個已安裝的外掛並重新探索。 |
| `/api/plugins/{id}/disable` | POST | token | 停用它，但不解除安裝。 |
| `/plugins/{id}/frontend/{path}` | GET | open | 當已啟用外掛的 manifest 宣告 `[frontend]` 時，提供其 `frontend/` 目錄中的檔案；否則回傳 404。route 會在每個 request 重讀 lockfile，因此安裝、啟用、停用與解除安裝不需重啟即可生效。`Cache-Control: no-cache` 會要求瀏覽器在更新後重新驗證。 |
| `/plugins/{id}/assets/{path}` | GET, HEAD | open | 提供已啟用外掛 `assets/` 目錄中的檔案，使用偵測到的 media type；未知時使用 `application/octet-stream`。此 route 使用與 frontend route 相同的每次 request lockfile 檢查與重新驗證。目錄 request 與外掛目錄以外的路徑都會被拒絕。 |

:::note 安裝外掛只能從本機操作
`inspect`、`install`、job `cancel`、`update` 與 `DELETE` 需要上述回送位址閘門。這些操作可以下載、安裝或移除第三方程式碼，檢查來源時也會依呼叫端提供的內容連線 GitHub。`reload`、`enable` 與 `disable` 只處理伺服器上已有的程式碼，因此只需要 token。這些 route 的拒絕回應使用 `busy`、`already_installed`、`consent_required` 等可供機器判讀的 `code`；請參閱[外掛中心](/advanced/plugins#plugin-center)下的表格。步驟名稱與失敗訊息來自共用的安裝實作，在所有 client 中都維持英文。
:::

## 選用套件包 {/* #optional-packs */}

| 端點 | 方法 | 驗證 | 說明 |
|----------|--------|------|-------------|
| `/api/packs` | GET | open | 列出所有套件包、已安裝項目、下載大小，以及此伺服器是否支援安裝，另包含 `active_job`、`remote_install_allowed`、`launch_mode`、`restart_available` 與 `gpu`。 |
| `/api/packs/{id}/install` | POST | token+loopback | 啟動安裝 job——回傳 `202 {"job_id"}`；套件包與外掛合計同時只執行一個 job。body（選用；未知 key 會回傳 422）：`items`（預設為整個套件包扣除已下載的內容）、`mode`——`live`（預設）或 `restart`，後者會讓 helper 在伺服器自行停止後安裝套件（參閱[讓伺服器重新啟動的安裝](/usage/optional-packs#讓伺服器重新啟動的安裝)）——以及 `variant`（僅 GPU 套件包：選哪個 torch wheel；不在 allowlist 的名稱會回傳 422）。在 job 建立前就會拒絕的情形：未知項目回傳 400；缺少必要套件包時回傳 `{detail, blocked_by}`；另一個安裝執行中時回傳 409 `{detail, job_id[, reason]}`；restart mode 無法使用或被拒絕時回傳 409 `{detail, command[, reason]}`（請自行執行 `command`）；磁碟容不下下載內容時回傳 507 `{detail, needed, free}`；restart helper 無法啟動時回傳 500。 |
| `/api/packs/jobs/{job_id}/cancel` | POST | token+loopback | 取消執行中的 job。進行中的下載會停止，不會完成目前檔案。由於 graph 與套件包下載共用一個 transfer session，取消操作也會中斷 graph 當時正在進行的 Hugging Face 資料集或 tokenizer 下載。 |
| `/api/packs/jobs/{job_id}/events` | GET | open | 回傳 job 在 `?cursor=` 之後的 log 與進度事件。`?wait=` 最多長輪詢 60 秒。 |
| `/api/packs/{id}/items/{item_id}` | DELETE | token+loopback | 刪除一個已下載的模型並釋放其位元組——未知項目回傳 404；該套件包自己的安裝執行中時回傳 409 `{detail, job_id}`；其他行程持有檔案而未能刪除時（Windows），則回傳 `removed: false`。套件包的 Python 套件無法從執行中的伺服器移除——參閱 `cdui packs remove`。 |

回送位址閘門用來保護會在伺服器 Python 環境中執行 package manager 的操作。型錄會限制可要求安裝的套件。型錄、restart mode 與 `cdui packs` 請見[選用套件包](/usage/optional-packs)。

## 檔案、模型、影像與媒體 {/* #files-models-images-and-media */}

| 端點 | 方法 | 驗證 | 說明 |
|----------|--------|------|-------------|
| `/api/files` | GET | open | 列出已上傳的資料檔（`.csv`、`.tsv`、`.txt`、`.json`）——CSVReader 等節點背後的 `DATA_FILE` 下拉選單。 |
| `/api/files/upload` | POST | token | 上傳資料檔（只接受上述四種副檔名）。 |
| `/api/files/download/{filename}` | GET | open | 下載資料檔。 |
| `/api/files/{filename}` | DELETE | token | 刪除資料檔。 |
| `/api/models` | GET | open | 列出已上傳的模型檔案。 |
| `/api/models/upload` | POST | token | 上傳模型權重檔。 |
| `/api/models/download/{filename}` | GET | open | 下載模型權重檔（支援巢狀路徑）。 |
| `/api/models/{filename}` | DELETE | token | 刪除模型檔案。 |
| `/api/images` | GET | open | 列出已上傳的影像檔案。 |
| `/api/images/upload` | POST | token | 上傳影像檔案。 |
| `/api/images/download/{filename}` | GET | open | 下載影像檔案。 |
| `/api/images/{filename}` | DELETE | token | 刪除影像檔案。 |
| `/api/media` | GET | open | 遞迴列出 run 產生的媒體（`.mp4`、`.webm`、`.gif`、`.png`、`.jpg`）。 |
| `/api/media/{filename}` | GET | open | 使用真正的 `Content-Type` 與 Range 支援，以 inline 形式提供單一媒體檔，因此 `<video>` 元素可以 seek。唯讀：檔案只會因節點（VideoWrite）寫入而出現在這裡——沒有 upload，也沒有 delete。 |

## LLM 代理 {/* #llm-proxy */}

| 端點 | 方法 | 驗證 | 說明 |
|----------|--------|------|-------------|
| `/api/llm/chat` | POST | token | 從設定的 provider 串流統一格式的 SSE chat completion（OpenAI / OpenRouter / Anthropic / OpenAI-Codex / 自訂 OpenAI 相容端點）。 |
| `/api/llm/models` | POST | token | 列出某個 provider 可用的模型。 |
| `/api/llm/codex/login` | POST | token | 啟動 OpenAI-Codex（ChatGPT 帳號）OAuth 登入流程。 |
| `/api/llm/codex/status` | GET | open | 回報 OpenAI-Codex OAuth 登入狀態。 |
| `/api/llm/codex/logout` | POST | token | 清除已儲存的 OpenAI-Codex OAuth token。 |

## 限制與錯誤 {/* #limits-and-errors */}

- **Body 大小。** `MAX_RUN_BODY_BYTES`（64 MB，`CODEFYUI_MAX_RUN_BODY_BYTES`）限制每個 request body，並在位元組抵達時逐步計數，包括 chunked request。四條 route 改用 `MAX_UPLOAD_SIZE`（500 MB，`CODEFYUI_MAX_UPLOAD_SIZE`）：`/api/files/upload`、`/api/images/upload`、`/api/models/upload` 與 `/api/custom-nodes/upload`。這些 route 另保留 64 KB 給 multipart metadata，並將設定的上限套用到檔案本身。超過任一上限時會回傳 413。
- **拒絕順序。** Host guard 先執行，可能回傳 421；接著是驗證，可能回傳 403 或 401；最後才檢查 body 大小。被拒絕的 request body 不會被讀取。因此，未驗證的 request 不會收到 413 response。
- **WebSocket frame。** Transport 會執行 `WS_MAX_MESSAGE_BYTES`（`CODEFYUI_WS_MAX_MESSAGE_BYTES`，預設等於 request body 上限）。frame 超過上限時，連線會以 code 1009 關閉，而不是回傳 413。手動啟動 uvicorn 時請參閱[把 graph 當成函式呼叫，第 8 節](/usage/graph-as-a-function#8-limits-and-gotchas)。
- **錯誤格式。** API 依 route 類別使用三種格式：
  1. `{"detail": "<text>"}`——預設格式，包含 413 response。套件包 route 會在 `detail` 旁加入 `job_id`、`reason`、`blocked_by`、`command`、`needed` 與 `free` 等欄位。
  2. `{"detail": {"code", "message", "details"}}`——應用程式管理 route 與 `/api/keys`，例如 404 `app_not_found`、422 `incomplete_cursor` 與 404 `key_not_found`。
  3. `{"detail": {"code", ...}}`，不含 `message`——外掛中心 route。其他欄位可能包含 `job_id`、`known`、`missing_capabilities`、`allowed_modules`、`plugin_id`、`inspection_id` 與 `id`。

  `POST /api/graph/run/{name}` 與 `POST /api/apps/{slug}/invoke` 是三者共同的例外：每一個 response——包括 413 與 401——都是九鍵 run envelope，參閱[這份 envelope](/usage/graph-as-a-function#the-response-envelope)。

## WebSocket 協定 {/* #websocket-protocol */}

| 端點 | 方法 | 驗證 | 說明 |
|----------|--------|------|-------------|
| `/ws/execution` | WebSocket | token | run 的附掛／訂閱檢視：`execute` 啟動一個 run，`attach` 從指定 cursor 重播事件 log 再接著即時追蹤，`detach` 取消訂閱，`cancel` 停止 run。關閉 socket 永遠不會取消。 |

**交握。** `ws://<host>:<port>/ws/execution?token=<session token>`——由於瀏覽器無法在 WebSocket 交握時設定 header，因此使用 `?token=` query parameter；非瀏覽器 client 也可使用 `X-CodefyUI-Token` header。在 socket 被接受前的 close code：`Host`——或瀏覽器的 `Origin`——不在白名單時為 **4003**；token 遺漏或無效時為 **4401**；之後 frame 超過大小上限時為 **1009**。

**Client action**——帶有 `action` 欄位的 JSON text frame：

| 動作 | 欄位 | 效果 |
|--------|--------|--------|
| `execute` | `nodes`、`edges`、`presets`、`subgraphs`、`device`、`seed`、`deterministic`、`record_outputs`、`changed_nodes`，以及畫布旗標（`verbose_mode`、`graph_id`、`weights_persistent`、`backward_mode`、`auto_backward`、`error_mode`、`max_retries`） | 在 interactive lane 送出，並附掛到新的 run。此 socket 已在追蹤的 run 會被 detach，而不是 cancel。 |
| `attach` | `run_id`、`cursor`（整數、至少為 0，且不得超過 run 的最新 cursor） | 從 `cursor` 重播事件 log，接著即時追蹤；取代先前的 attachment。 |
| `detach` | — | 停止追蹤。絕不會取消。 |
| `cancel` | `run_id`（選用；預設為附掛的 run） | 協作式停止。`stop` 是 v2 之前的別名。 |
| `clear_cache` | — | 清除這個 socket 的 execution cache。 |

**Server frame。** 已儲存的 run 事件使用 `{type, run_id, cursor, ...payload}`（`execution_start`、`node_status`、`execution_stopped` 等）。client 重新附掛時使用 `cursor`。僅由 transport 產生的 frame 包含 `attached {run_id, cursor, status}`、`detached {run_id}`、`cancel_ack {run_id, status, cancelled}`、沒有可取消 run 時的 `execution_stopped {reason: "not_running"}`、`cache_cleared`、action 無效或未知時的 `error {error}`，以及 run service 拒絕送出時的 `execution_error {error}`。`cancel_ack` 只確認已收到 request；run 停止後會再發出自己的 `execution_stopped` 事件。因 interactive 上限或每個 session 一個 run 的規則而拒絕時，也會包含 `rejected: true` 與目前附掛的 `run_id`。此時不會啟動新 run，原有 attachment 會維持作用中。
