---
sidebar_position: 8.5
title: 選用套件包
description: 安裝特定 LLM 節點與 GPU 後端所需的選用 Python 套件和模型檔。
---

# 選用套件包

選用套件包內含基本 CodefyUI 安裝未提供的 Python 套件和模型檔。型錄內容皆採用寬鬆授權，套件版本固定在已測試的範圍。基本安裝可保持精簡並離線運作；只需安裝圖所需的套件包。

你可以從**套件中心**安裝套件包（工具列 > 設定 > 選用套件，或側邊欄的**自訂與外掛**分頁 > **選用套件** > **套件中心...**），也可以執行 `cdui packs install <id>`。套件中心會顯示各項下載大小和進度。兩種方式使用相同的安裝程式與型錄。型錄是白名單，只接受預先定義的 id；請求內容無法將 pip spec、repository id 或 URL 傳給安裝程式的子行程。

:::note 執行圖時不會下載套件內容
如果缺少必要的套件包，**執行**會在該節點停止並指出套件，不會下載內容。`TextCorpusDataset`、`HuggingFaceDataset` 和 `Tokenizer` 可以自行從 Hugging Face Hub 取得小型資產，並使用各自的快取。此限制只適用於套件中心管理的內容。
:::

## 為什麼是選用的

基本安裝可以離線啟動及執行。`WordVector` 預設使用 `demo-16d`，這是內建的手工詞彙表，包含 59 個詞和 16 個可解讀維度。它不需下載，且向量經過設計，使 `king - man + woman = queen` 精確成立。

安裝或移除套件包只會變更相關選項的可用狀態，不會變更基本安裝的其他部分。缺少下載內容的 `select` 選項會變灰並提供安裝操作。如果該選項是圖中儲存的目前值，它仍可選取並會顯示警告，避免開啟面板時變更儲存值。所有後端都需要同一套件的節點（包括 `TextEmbedding` 和 `HFTextGenerate`）會在節點層級顯示標記。移除套件後，缺少內容的標記會再次出現。

## 套件目錄

| 套件包 | 內容 | 下載量 | 授權 | 可用功能 |
|--------|------|--------|------|----------|
| `sentence-embeddings` | `sentence-transformers` 套件和四個小型編碼器：`all-MiniLM-L6-v2`、`paraphrase-multilingual-MiniLM-L12-v2`、`bge-small-zh-v1.5`、`multilingual-e5-small` | 每個模型分別為 90 MB、470 MB、95 MB、470 MB（它們是替代選項，不是一組），另加 pip 套件 | 兩個 MiniLM 模型為 Apache-2.0；`bge` 和 `e5` 為 MIT | 整個 `TextEmbedding` 節點，以及 `WordVector` 的四個句子後端 |
| `word-vectors` | `glove-wiki-gigaword-50.gz`，包含 400,000 個詞、50 個維度的 GloVe 詞向量表；不含 Python 套件 | 69 MB，另加儲存在旁邊、約 83 MB 的轉換表 | PDDL-1.0 | `WordVector` 的 `glove-50d` 後端 |
| `rag` | `Qwen2.5-0.5B-Instruct`，可在 CPU 上執行的本機生成模型 | 約 1 GB | Apache-2.0 | `HFTextGenerate`；檢索鏈需先安裝 `sentence-embeddings` |
| `gpu-torch` | 為這台機器選取的 CUDA 或 ROCm PyTorch 版本 | 依版本而異 | PyTorch 的 BSD-3-Clause 授權 | 不新增節點。請使用 `cdui install --gpu <variant>` 安裝，而非 `cdui packs`；詳見 [GPU 與裝置設定](../getting-started/gpu-device.md) |

下載量欄列出網路傳輸大小。安裝 `word-vectors` 會下載 69 MB，並寫入另一份約 83 MB 的轉換表。磁碟預檢查需要約 230 MB 可用空間；移除時會刪除這兩個檔案。

`rag` 套件依賴 `sentence-embeddings` 的 Python 套件，但此相依性不會安裝編碼器模型。完全在本機執行的 RAG 圖還需要 `multilingual-e5-small` 等編碼器，合計下載量約 1.5 GB。請先安裝編碼器。由於 `rag` 本身不含 Python 套件，在 `sentence-embeddings` 函式庫可匯入之前，系統會拒絕安裝。若在基本安裝上執行 `cdui packs install rag`，指令會以離開碼 `2` 結束，顯示 `RAG stack needs another pack first`，並指出必要套件。套件中心也會執行相同檢查。請依序執行：

```bash
cdui packs install sentence-embeddings --items multilingual-e5-small
cdui packs install rag --yes
```

在套件中心內，也請先安裝編碼器項目，再安裝 RAG 模型。可以在課程使用前先完成兩項安裝。

## 安裝與移除

**在應用程式中。** 從工具列 > 設定 > 選用套件開啟套件中心。每個套件包會列出項目、大小和下載狀態。選取項目並開始安裝後，可以查看記錄和位元組計數器。**取消安裝**會停止目前的傳輸。模型下載會從部分檔案續傳；GloVe 詞向量表等單一檔案資產則會重新下載。同一時間只能執行一個安裝工作。

**從終端機。** CLI 使用相同的安裝程式與型錄：

```bash
cdui packs list                                       # 每個套件、其中的項目、大小與授權
cdui packs status                                     # 同上，加上這個 venv 的 PyTorch 版本與下一步
cdui packs install sentence-embeddings --items all-MiniLM-L6-v2
cdui packs install word-vectors --yes
cdui packs remove word-vectors glove-50d
```

`--items a,b` 只下載列出的項目；省略時會下載套件內所有缺少的項目。`--yes` 會略過下載大小確認，在無終端機的環境中（包括 CI 和管線命令）為必要選項。只接受型錄中的 id。離開碼請參閱[套件包指令](../getting-started/cli-commands.md#套件包指令)。在開發 checkout 中，`uv pip install -e ".[llm-sentence]"` 會安裝 `sentence-embeddings` 使用的套件版本，但不會下載模型。

**檔案位置。** 由 `cdui start` 或 `cdui dev` 啟動的伺服器使用 `<install dir>/.codefyui_dev/cache/`；預設安裝目錄為 `~/CodefyUI`。同一個使用者資料根目錄會儲存 session token 和 plugin lockfile。詳見[專案目錄](./project-directories.md#6-建立-api-keyinvoke-需要)。手動啟動的 `uvicorn app.main:app` 行程則使用 Windows 的 `%LOCALAPPDATA%\codefyui\Cache`、macOS 的 `~/Library/Caches/codefyui` 或 Linux 的 `~/.cache/codefyui`。Hugging Face snapshot 儲存在 `hf/` 下，例如 `hf/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/<revision>/`。GloVe 詞向量表等單一檔案下載會儲存在快取根目錄。`packs/state/` 下的 JSON 檔會記錄各項目已完成，避免將部分 snapshot 視為完整下載。如果設定 `CODEFYUI_USER_DATA_DIR`，快取會位於 `<dir>/cache`，控制檔則位於 `<dir>/packs`。CodefyUI 的套件操作不會讀寫 `HF_HOME`。

**移除項目。** 執行 `cdui packs remove <pack> <item>` 或使用項目的刪除按鈕。這會刪除下載內容、轉換後的 GloVe `.npz` 等衍生檔案，以及狀態紀錄。Python 套件會保留，因為執行中的伺服器無法安全地從自己的直譯器移除套件。指令會改為印出解除安裝命令：

```text
uv pip uninstall --python <path-to-venv-python> sentence-transformers
```

請在伺服器停止時執行該命令。

**透過網路安裝。** 所有會修改狀態的 `/api/packs` 路由都要求伺服器綁定至 loopback，因為安裝會針對提供服務的直譯器執行套件管理程式。若伺服器是刻意對區網開放，可設定 `CODEFYUI_ALLOW_REMOTE_PACK_INSTALL=1` 允許安裝。兩種情況都會套用型錄白名單。詳見 [API 參考](../advanced/api-reference.md)。

### 讓伺服器重新啟動的安裝

部分安裝必須取代伺服器已匯入的套件。**GPU 版 PyTorch** 一律需要此模式。如果線上安裝的 constraints 偵測到 resolver 衝突，並在取代任何套件前停止，該套件也需要此模式。由 `cdui start` 管理的伺服器會記錄請求、啟動分離的輔助程式，然後關閉。輔助程式會等待伺服器結束、執行安裝、記錄結果，再使用原本的 `cdui start` 引數啟動伺服器。輔助程式不使用線上安裝的 constraints 檔，因此可在套件需要不同版本時取代 torch。

**套件中心行為。** **GPU 版 PyTorch** 卡片會顯示偵測到的 GPU 和已安裝版本；如果建議版本不同，也會顯示建議版本。控制列包含版本選單和**安裝並重新啟動**。說明文字為「有圖在執行時不會開始。」對應的命令仍可在**手動安裝指令**下查看。無法使用重新啟動模式時（包括使用 `cdui dev`），卡片不會顯示按鈕，而會顯示命令和無法使用的原因。

確認「要安裝 cu128 並重新啟動伺服器嗎？」等提示後，**伺服器重新啟動中**遮罩會阻擋頁面並顯示經過秒數。另一個伺服器行程開始回應時，頁面會重新載入，接著顯示「伺服器已重新啟動，GPU 版 PyTorch 可以使用了。」或「伺服器已重新啟動，但安裝 GPU 版 PyTorch 失敗：」，並附上原因或安裝程式的最後輸出。如果線上安裝因 resolver 衝突而停止，其活動橫幅會提供**重新啟動伺服器並安裝**。

重新啟動模式只會安裝 torch wheel 或套件的 Python 套件，不會下載模型項目，因為分離的輔助程式不包含應用程式的下載器。如果所選套件也包含模型，請在重新啟動後執行一般套件安裝。

**提供與拒絕條件。** `GET /api/packs` 會回傳 `restart_available`，只有值為 `true` 時才會顯示按鈕。重新啟動模式僅在以下條件全部成立時可用：

- 伺服器由 `cdui start` 啟動。`cdui dev` 行程無法自行重新啟動，因此會改為顯示手動命令。
- 啟動器仍位於原本的路徑。
- `CODEFYUI_ENABLE_RESTART_INSTALL` 不為 `0`。設為 `0` 會停用該機器上的重新啟動安裝，並顯示手動命令。

即使重新啟動模式可用，下列任一狀況仍會在寫入狀態或安裝套件之前拒絕請求：有圖正在執行或排隊、另一個安裝正在執行，或另一個重新啟動安裝仍在等待。torch wheel 的輔助程式需要虛擬環境所在磁碟區有 3 GB 可用空間；套件的 Python 套件則需要 1 GB。空間不足會建立失敗工作紀錄，不會進行部分安裝。即使原始命令是 `cdui start -f`，輔助程式也一律以背景模式重新啟動伺服器，因為沒有主控台可供附加。

**重新啟動狀態檔。** 受管理的伺服器將重新啟動狀態儲存在 `<install dir>/.codefyui_dev/`；如果設定 `CODEFYUI_USER_DATA_DIR`，則儲存在 `<dir>`。pending 認領紀錄會指出要求的安裝和輔助程式行程。

輔助程式執行期間，認領狀態為**收尾中**（finishing）。輔助程式尚未記錄行程 ID 時，建立未滿 60 秒的認領紀錄也視為收尾中。在此狀態下，`cdui start` 不會在輔助程式修改虛擬環境時啟動第二個伺服器，而會要求查看 `cdui status`。`cdui update` 和 `cdui dev` 也會拒絕並回傳離開碼 `1`；`cdui start` 回傳 `0`。如果發出認領紀錄的伺服器仍在執行，且紀錄建立未滿 15 分鐘，套件中心也會拒絕另一個重新啟動安裝。

輔助程式結束後，或 60 秒內都未記錄行程 ID 時，認領狀態為**已中斷**（abandoned）。`cdui start` 會刪除已中斷的認領紀錄並正常啟動。伺服器也會在啟動時清除其認領紀錄；新的重新啟動安裝可取代已中斷的紀錄。

每個使用者資料根目錄只支援一個受管理伺服器的一份重新啟動認領紀錄。不支援兩個受管理伺服器共用同一根目錄，例如同時執行前景的 `cdui start -f` 和背景的 `cdui start`。請為第二個伺服器設定不同的 `CODEFYUI_USER_DATA_DIR`。

```text
<user data>/packs/pending_restart.json      要求的安裝和輔助程式狀態
<user data>/packs/last_restart_job.json     重新載入的頁面和 cdui status 在一小時內讀取的結果
<user data>/packs/logs/restart-<job>.log    完整安裝程式輸出
```

**如果伺服器沒有恢復。** 如果原始伺服器在 30 秒後仍回應，遮罩會顯示「伺服器沒有重新啟動。請執行下面的指令後重新載入：」。如果 10 分鐘內沒有伺服器恢復，則顯示「等了 10 分鐘，伺服器還沒恢復。」兩種狀態都會顯示 API 提供的命令（如果有），並提供**立即重新載入**。關閉遮罩或等待逾時不會停止輔助程式或刪除結果紀錄。

請執行 `cdui status` 查看重新啟動狀態。其中的「重啟安裝」列會指出套件；輔助程式執行時顯示**收尾中**，結束但未清除認領紀錄時顯示**已中斷**。完成後的一小時內，「上次重啟安裝」會顯示記錄的結果。如果安裝成功但重新啟動失敗，紀錄會保留安裝狀態、加入 `relaunch: failed` 並包含記錄檔路徑；`cdui status` 會將整體重新啟動標示為失敗。完整安裝程式輸出位於 `packs/logs/restart-<job>.log`。

認領狀態為**已中斷**時，請執行 `cdui start` 刪除紀錄並啟動伺服器。狀態為**收尾中**時，`cdui start` 會拒絕啟動另一個行程，並要求查看 `cdui status`。

## 畫布上會看到的變化

缺少套件時，編輯器會在執行前顯示狀態。所有後端都需要套件的 `TextEmbedding` 和 `HFTextGenerate` 會在節點面板顯示**需要套件**標籤，但仍可拖曳。放在畫布上的節點會顯示**需套件**徽章，點擊後會開啟套件中心並定位至必要套件。`WordVector` 中無法使用的 **backend** 選項會變灰，`demo-16d` 仍可選取，欄位下方則顯示**安裝套件**連結。如果使用不含套件中心的版本，請執行 `cdui packs list` 查看可用狀態。

執行到缺少必要內容的節點時，會在該節點停止並指出需求：

```text
Model 'all-MiniLM-L6-v2' from the Sentence embeddings pack is not downloaded. Open Package Center (toolbar > Settings > Optional packs) to download it; graph runs never download (pack=sentence-embeddings)
```

`(pack=<id>)` 後綴可供機器解析。編輯器會擷取 id，顯示錯誤通知，並提供聚焦至必要套件的**開啟套件中心**按鈕。該次執行不會取得套件內容。

## 套件相關節點參考

### WordVector

`WordVector` 會從查找表或編碼器，為每個輸入詞回傳一個向量。請依所需表示方式選取後端：

| 後端 | 需要 | 行為 |
|------|------|------|
| `demo-16d` | 無 | 內建 59 個詞和 16 個手工維度，涵蓋王室、神性、性別、動物類別、移動、交通工具、食物和天氣。標準類比題依設計會得到精確結果。 |
| `glove-50d` | `word-vectors` | 包含 400,000 個詞的 GloVe 詞向量表。標準類比題會得到近似結果。 |
| `sentence-transformers/all-MiniLM-L6-v2` 和另外三個模型 id | `sentence-embeddings` 中所選的模型 | 每次套用至一個詞的句子編碼器。這些模型以句子訓練，也用於檢索系統。 |

`normalize` 會對每列套用 L2 正規化，使下游內積等於餘弦相似度。`keep_oov` 會為查找表中不存在的詞回傳零向量，而非省略該詞。此設定只適用於 `demo-16d` 和 `glove-50d`；句子編碼器會為任何字串回傳向量。

**已淘汰的後端名稱。** 早期預覽版儲存的圖可能包含 `glove-100d` 或 `minilm-sentence-384d`。這些值會回傳錯誤並指出替代值，無法透過下載恢復。請分別改用 `glove-50d` 和 `sentence-transformers/all-MiniLM-L6-v2`。

### TextEmbedding

`TextEmbedding` 會為每段輸入文字回傳一個稠密向量。語意搜尋和 RAG 圖會用它嵌入文件與問題，以供比較。此節點需要 `sentence-embeddings`。

請連接 `texts`（例如切塊器輸出的清單）或 `text`（單一字串）。同時連接兩者無效。主要參數如下：

- **`model`** 可選取下列四個編碼器之一。
- **`prefix`** 會加在每段文字之前。`multilingual-e5-small` 要求問題使用 `query: `、文件使用 `passage: `；其他三個模型不需這些前綴。
- **`split_lines`** 預設啟用，會分別編碼每個非空白輸入行。停用後，會將多行文件編碼為一個向量。
- **`max_seq_length`** 設定每段文字的 token 上限。`0` 使用模型預設值：paraphrase-multilingual 為 128、all-MiniLM 為 256、bge 和 e5 為 512。較長文字會被截斷。
- **`normalize`** 預設啟用。其他控制項為 **`batch_size`**、**`label_chars`** 和 **`device`**。

`embeddings` 和 `labels` 輸出可連接至 `CosineSimilarity` 與 `EmbeddingScatter`。[範例集](./examples-gallery.md)中的 **Sentence Similarity (zh-TW)** 使用此路徑並需要該套件。

### RAG 鏈

檢索增強生成圖包含七個節點，其中兩個需要下載：

```text
DocumentLoader -> TextChunker -> TextEmbedding -> VectorStore -> Retriever -> PromptBuilder -> HFTextGenerate
                                                                                           (或 LLMChat)
```

| 節點 | 行為 | 需要 |
|------|------|------|
| `DocumentLoader` | 讀取目錄中的每個 `.md` 和 `.txt` 檔，並為每個檔案回傳 `{text, source}`。不支援 PDF、HTML 和 DOCX。`recursive` 會包含子目錄，`max_docs` 會限制檔案數。將 `source` 設為 `uploaded_file`，可讀取透過 `file` 選取的單一 `.txt`。 | 無 |
| `TextChunker` | 切分文件以供嵌入和建立提示。`characters` 使用固定視窗，不依賴詞界。`sentences` 和 `paragraphs` 使用作者定義的界線，並合併單位至 `chunk_size`。每個切塊都包含來源、`start_char` 和 `end_char`；`text[start_char:end_char]` 等於切塊內容。 | 無 |
| `TextEmbedding` | 為每個切塊和問題建立向量。兩次使用必須選取相同模型。 | `sentence-embeddings` |
| `VectorStore` | 將切塊向量與文字及 metadata 儲存在一個 `[N, D]` 矩陣。每列會正規化為單位長度，因此餘弦搜尋只需一次矩陣乘法。資料儲存在記憶體，可從快取的嵌入重建。 | 無 |
| `Retriever` | 計算問題和每列的分數，回傳高於 `min_score` 的前 `top_k` 列，並包含各切塊的來源。記錄會列出每個結果的分數。 | 無 |
| `PromptBuilder` | 將檢索到的切塊和問題插入限制答案只能依據該脈絡的範本。範本必須包含 `{context}` 和 `{question}`。將 `TextInput` 連接至 `template` 可提供自訂範本。 | 無 |
| `HFTextGenerate` | 在本機執行 Qwen2.5-0.5B-Instruct、套用其 chat template，並依 token 回報生成進度。 | `rag` |

可以在鏈尾用 `LLMChat` 取代 `HFTextGenerate`。它會將相同提示傳給 Ollama 或託管提供者，且不需選用套件。

**隨附語料。** `backend/data/samples/rag` 包含五份關於 CodefyUI、節點與連線、訓練、嵌入與 RAG，以及選用套件的短筆記。每份筆記都有英文和繁體中文區段。因此，範例不需設定語料即可執行，也可示範多語言編碼器。若要使用其他內容，請將 `DocumentLoader.directory` 設為包含 `.md` 和 `.txt` 檔的目錄。

**e5 前綴。** `multilingual-e5-small` 的訓練資料會在問題前加上 `query: `，在索引文件前加上 `passage: `。兩個 `TextEmbedding` 節點必須使用不同前綴和相同 `model`。不同模型的嵌入位於不同向量空間，但圖無法偵測此不相符；檢索仍會回傳 `top_k` 個結果和無效的相似度分數。

**僅依脈絡回答。** 本機 0.5B 模型只具備少量 CodefyUI 既有資訊。範例筆記會透過提示提供資訊，不需微調。`PromptBuilder` 會要求模型只依檢索到的脈絡回答。除非 `min_score` 濾除結果，`Retriever` 一律回傳最接近的切塊，即使語料中沒有答案。因此，拒絕不受支援答案是由提示中的指示控制。

**CPU 效能。** 在筆電 CPU 上，生成速度通常為每秒數個 token，因此一個答案可能需要數秒至數十秒。第一次執行還可能需要數秒從磁碟載入權重。隨附問題通常會在 160 個 token 上限前完成。這些數字依模型大小估算，並非 benchmark 實測。節點會回報每個 token 的進度。GPU 速度較快；除非節點覆寫設定，`device` 會遵循全域選擇。

兩個圖都列於[範例集](./examples-gallery.md)，且各自的範例目錄都有 `README.md`。**RAG, fully local**（`examples/LLM/RAG-Local-Offline`）需要 `qwen2.5-0.5b-instruct` 和 `multilingual-e5-small`，不會向提供者發出請求。**RAG with a chat API**（`examples/LLM/RAG-LLMChat-API`）使用相同檢索節點，並以 `LLMChat` 取代最後一個節點；它需要編碼器，以及 Ollama 或提供者金鑰。兩個圖使用相同問題時，檢索脈絡會保持相同，方便比較生成器。

## 如何挑選嵌入模型

| 模型 | 語言 | 維度 | token 上限 | 需要前綴 | 下載量 |
|------|------|-----:|-----------:|----------|-------:|
| `sentence-transformers/all-MiniLM-L6-v2` | 英文 | 384 | 256 | 否 | 90 MB |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`（預設） | 50 種以上，包含繁體中文 | 384 | 128 | 否 | 470 MB |
| `BAAI/bge-small-zh-v1.5` | 中文 | 512 | 512 | 否 | 95 MB |
| `intfloat/multilingual-e5-small` | 100 種以上，包含繁體中文 | 384 | 512 | `query: ` / `passage: ` | 470 MB |

- **只使用英文且要減少下載量** — `all-MiniLM-L6-v2`；也是四個模型中最快的選項。
- **使用繁體中文或混合語言** — 預設的 `paraphrase-multilingual-MiniLM-L12-v2`。它不需前綴，並可對齊不同語言中的同義文字。
- **只使用中文且有較長段落** — `bge-small-zh-v1.5`。每段支援 512 個 token，預設模型則為 128；下載量為 95 MB。
- **分別嵌入問題和文件以供檢索** — `multilingual-e5-small`，問題使用 `query: `，文件使用 `passage: `。省略前綴會降低檢索品質，但不會產生錯誤。

行程最多同時將兩個模型保留在記憶體中。載入第三個模型時，會移除最久未使用的模型。

## 疑難排解

- **「reports installed but `sentence_transformers` cannot be imported」** — 狀態紀錄存在，但目前的直譯器無法匯入套件。請從套件中心重新安裝 `sentence-embeddings`，或執行 `cdui packs install sentence-embeddings --yes`。
- **「Model ... is not downloaded」** — Python 套件已安裝，但所選模型尚未下載。四個編碼器模型需分別安裝。請在套件中心安裝指定模型，或執行 `cdui packs install sentence-embeddings --items multilingual-e5-small`。
- **CPU 編碼速度慢。** 這些模型包含 22M 至 118M 個參數，並支援 CPU 執行。第一次編碼可能需要數秒載入權重；之後的小批次句子通常不到一秒。GloVe 需進行一次文字轉 `.npz` 作業，約需數秒並會回報進度；之後每個行程載入約需一秒。
- **生成速度慢。** `HFTextGenerate` 會逐一解碼 token。0.5B 模型在筆電 CPU 上通常每秒產生數個 token，因此較長答案可能需要數十秒。若要縮短時間，可降低 `max_new_tokens`；降低 `Retriever.top_k` 或 `PromptBuilder.max_context_chars`；或在可用時將 `device` 設為 `cuda`。以 `LLMChat` 取代最後一個節點，可將生成工作交給 Ollama 或託管提供者。
- **答案忽略脈絡。** 請查看 `Retriever` 記錄中的各結果分數。最高分接近 0.3 通常表示語料不含答案。如果回傳切塊包含答案，但提示缺少所需段落，請提高 `top_k`。如果沒有回傳切塊，請降低 `min_score`；`0` 會保留所有切塊。沒有脈絡時，`PromptBuilder` 會寫入 `(no context retrieved)` 並發出警告。另請確認兩個 `TextEmbedding` 節點使用相同模型；不同模型可能產生看似合理但無效的檢索結果，且不會報錯。
- **Windows 路徑。** Hugging Face snapshot 路徑的層級很深。請啟用 Windows 長路徑支援，或將 `CODEFYUI_USER_DATA_DIR` 設為較短的路徑。如果移除操作回報項目已取消登錄，但另一個行程仍開啟檔案而使檔案留在磁碟上，請停止伺服器，再手動刪除目錄。
- **「cannot be installed while the server is running」** — 線上安裝使用 constraints 檔，將直譯器已載入的每個發行套件固定在現有版本。它可以新增套件，但不能取代已載入的套件。遇到衝突時，安裝會在取代任何內容前停止、回傳 CLI 離開碼 `3`，並印出 `uv pip install` 命令。請執行 `cdui stop`、執行該命令，再重新啟動。GPU PyTorch 一律需要重新啟動模式，並使用 `cdui install --gpu <variant>`，而非 `cdui packs install`。
- **安裝期間伺服器停止且未恢復。** 這只會發生於重新啟動模式。請執行 `cdui status`。如果「重啟安裝」顯示**收尾中**，請等待；`cdui start` 不會啟動第二個行程。如果顯示**已中斷**，或出現「上次重啟安裝」，請查看 `<user data>/packs/logs/restart-<job>.log`，再執行 `cdui start`；啟動時會清除過期的認領紀錄。重新啟動失敗會記錄為 `relaunch: failed`，不會取代安裝本身的狀態。套件中心只會在遮罩仍追蹤相同工作時顯示結果。遮罩逾時或之後才開啟分頁時，請查看保留一小時的「上次重啟安裝」列和記錄檔。
- **磁碟空間不足。** 系統會在下載前檢查空間。錯誤會列出所需空間和可用空間。

## 授權

型錄中的每個項目都採用寬鬆授權。`cdui packs list` 會顯示各下載項目的授權；型錄定義位於 `backend/app/core/packs/catalog.py`。

| 項目 | 授權 |
|------|------|
| `sentence-transformers`（Python 套件） | Apache-2.0 |
| `all-MiniLM-L6-v2` | Apache-2.0 |
| `paraphrase-multilingual-MiniLM-L12-v2` | Apache-2.0 |
| `bge-small-zh-v1.5` | MIT |
| `multilingual-e5-small` | MIT |
| `glove-50d`（glove-wiki-gigaword-50） | PDDL-1.0 |
| `qwen2.5-0.5b-instruct` | Apache-2.0 |

CodefyUI 可採用 AGPL-3.0 或商業授權；詳見[授權](../licensing.md)。套件內容保留各自的授權，並直接從各自的上游來源下載。CodefyUI 不會轉散布這些內容。
