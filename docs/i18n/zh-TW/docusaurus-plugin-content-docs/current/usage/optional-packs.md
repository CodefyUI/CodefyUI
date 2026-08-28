---
sidebar_position: 8.5
title: 選用套件包
description: 安裝精選的模型套件包，把 LLM 節點從玩具示範切換成真正的嵌入模型、GloVe 詞向量與本機生成模型。
---

# 選用套件包

套件包（pack）是一組精選過的 Python 套件與模型檔，預設安裝刻意不把它們放進來。基本安裝要小到可以直接發給一整間教室；一堂句子嵌入的課需要的那四百多 MB，等到真的有課要用時再下載。型錄裡的每一項都不大、授權都寬鬆，而且版本都鎖在這份程式碼實際測試過的範圍。

安裝套件包可以用應用程式裡的**套件中心**（工具列 > 設定 > 選用套件包），它會列出每個套件包要花多少空間，並用進度條顯示下載狀況；也可以在終端機用 `cdui packs install <id>`。兩者是同一套安裝流程、同一份型錄的兩個入口，而且這份型錄是一份白名單，不是套件管理器：兩邊都只接受型錄裡既有的 id，所以不會有任何 pip 安裝字串、repo id 或網址從請求內容一路傳進子行程。

:::note 一條到處適用的規則：執行圖形時永遠不會下載套件包內容
在缺少套件包的情況下按 **執行**，那個節點會停下來，並回報一句指名是哪個套件包的訊息，而且不會下載任何東西。四百多 MB 在執行途中開始下載、在教室的網路上、沒有進度條也沒辦法取消 —— 這不是一個「執行」按鈕可以做的事。至於本來就會自己去 Hugging Face Hub 抓小型資產的節點（`TextCorpusDataset`、`HuggingFaceDataset`、`Tokenizer`）不受影響：它們有自己的快取，這條規則講的是套件中心管的那一份。
:::

## 為什麼是選用的

預設安裝一開機就能離線使用，凡是不用下載就能教的課，就都不用下載。`WordVector` 的預設 backend 是 `demo-16d`：一份手工編寫、59 個詞、16 個可解讀維度的詞彙表，直接內建在程式裡，不需要下載，而且 `king - man + woman = queen` 算出來剛好精確成立 —— 因為這些向量本來就是為了讓它成立而寫的。它就是那個「玩具」，而這正是它存在的意義。

裝上套件包，除了讓某些選項亮起來以外，不會改變基本安裝的任何東西。下拉選單裡缺少下載的選項會被變灰，並附上安裝的入口；整個節點的所有 backend 都來自同一個套件包時（`TextEmbedding` 與 `HFTextGenerate`，這兩個節點都只能從套件包來），整個節點都會變灰。已經裝好的東西不論如何都照常運作，而移除套件包只會讓那些選項再變回灰色。

## 套件目錄

| 套件包 | 內容 | 下載量 | 授權 | 解鎖什麼 |
|--------|------|--------|------|----------|
| `sentence-embeddings` | `sentence-transformers` 套件，加上四個小型編碼器：`all-MiniLM-L6-v2`、`paraphrase-multilingual-MiniLM-L12-v2`、`bge-small-zh-v1.5`、`multilingual-e5-small` | 四個模型分別是 90 MB、470 MB、95 MB、470 MB（它們是互相替代的選項，不是一整套都要裝），另外還有 pip 套件 | 兩個 MiniLM 模型是 Apache-2.0，`bge` 與 `e5` 是 MIT | `TextEmbedding`（整個節點），以及 `WordVector` 的四個句子編碼 backend |
| `word-vectors` | `glove-wiki-gigaword-50.gz`：真正的 400,000 詞、50 維 GloVe 詞向量表。完全不需要 Python 套件 | 69 MB，另外會在旁邊產生約 83 MB 的一次性轉檔 | PDDL-1.0 | `WordVector` 的 `glove-50d` backend |
| `rag` | `Qwen2.5-0.5B-Instruct`，一個小到可以在 CPU 上跑的本機生成模型 | 約 1 GB | Apache-2.0 | `HFTextGenerate`，以及它帶來的「先檢索、再生成」整條鏈：不用連網，就能從你自己的筆記裡回答問題。需要先裝好 `sentence-embeddings` |
| `gpu-torch` | 符合這台機器的 CUDA 或 ROCm PyTorch 版本 | 依變體而異 | PyTorch 自己的授權（BSD-3-Clause） | 不會多出新節點；而是讓每個用得到加速器的節點都用得到。它根本不是用 `cdui packs` 裝的：請執行 `cdui install --gpu <variant>`，見 [GPU 與裝置設定](../getting-started/gpu-device.md) |

表格裡的大小是實際下載的量。GloVe 那一列是唯一一個「裝完會比下載的更佔空間」的：69 MB 的下載，加上安裝時在它旁邊寫下的 83 MB 轉檔表，所以磁碟空間預檢查會在開始前要求大約 230 MB 的可用空間；移除這個項目時，這些空間會一併還回來。

`rag` 這一列要下載的是兩樣東西，不是一樣。它相依於 `sentence-embeddings`，而這層相依帶來的是那個套件包的 Python 套件，**不是**任何一個編碼器：RAG 圖裡負責檢索的那一半，仍然需要一個模型來做嵌入。所以一堂完全在本機跑的 RAG 課，需要 `rag` 裡的 `qwen2.5-0.5b-instruct`，「加上」`sentence-embeddings` 裡的 `multilingual-e5-small` —— 兩者合計約 1.5 GB，而且要先裝編碼器那一個。`rag` 自己不帶任何 Python 套件，所以在 `sentence-embeddings` 的函式庫還匯入不了之前，它會被擋下來：在預設安裝上執行 `cdui packs install rag` 會以離開碼 2 結束，並回報 `RAG stack needs another pack first`，同時印出該先裝哪一個套件包；套件中心也會基於同一條規則拒絕這次安裝，因為兩者本來就是同一條規則的兩個入口。所以請先執行 `cdui packs install sentence-embeddings --items multilingual-e5-small` —— 這一個指令會先裝好 Python 套件、再下載那個編碼器 —— 之後再執行 `cdui packs install rag --yes`。在套件中心裡勾選這兩個項目也是同樣的順序：等 sentence-embeddings 的函式庫到位，RAG stack 才會解鎖。不論如何，下載和程式碼都還是兩個可以分開的決定，所以教室可以在上課前一天就把兩份都先抓好。

## 安裝與移除

**在應用程式裡。** 打開套件中心（工具列 > 設定 > 選用套件包）。每個套件包都會列出其中的項目、大小，以及是否已經下載；勾選你要的項目、開始安裝，過程中可以看到記錄與已下載的位元組數。**取消**會在傳輸到一半時停止。模型下載下次會從已下載的部分接續；像 GloVe 詞向量表這類資產檔案則會重新下載。同一時間只會有一個安裝工作在跑。

**在終端機裡。** 同一套安裝流程，走同一條程式碼路徑：

```bash
cdui packs list                                       # 每個套件包、其中的項目、大小與授權
cdui packs status                                     # 同上，再加上這個 venv 的 PyTorch 版本與接下來該做什麼
cdui packs install sentence-embeddings --items all-MiniLM-L6-v2
cdui packs install word-vectors --yes
cdui packs remove word-vectors glove-50d
```

`--items a,b` 只會下載指定的項目，預設則是補齊這個套件包缺的全部。`--yes` 會跳過下載大小的確認，在沒有終端可以確認的情況下（CI、把輸出接到管線）必須加上。只接受型錄裡的 id。給腳本用的離開碼列在 [套件包指令](../getting-started/cli-commands.md)。若你用的是開發用的 checkout，`uv pip install -e ".[llm-sentence]"` 裝的是和 `sentence-embeddings` 套件包完全相同的版本範圍；模型仍然要另外下載。

**檔案會放在哪裡。** 放在 CodefyUI 的資產快取裡：Windows 是 `%LOCALAPPDATA%\codefyui\Cache`，macOS 是 `~/Library/Caches/codefyui`，Linux 是 `~/.cache/codefyui`。Hugging Face 的 snapshot 放在 `hf/` 底下（`hf/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/<revision>/`），像 GloVe 詞向量表這種單一檔案則直接放在快取根目錄，另外每個項目會在 `packs/state/` 留一個小小的 JSON，記錄「這個下載真的完成了」—— 因為在磁碟上，一份只下載到一半的 snapshot 看起來和完整的一模一樣。設定 `CODEFYUI_USER_DATA_DIR` 會把這些全部搬走：快取變成 `<dir>/cache`，控制檔變成 `<dir>/packs`。這裡不會去讀或寫 `HF_HOME` —— 那是整台機器共用、和你其他工具共享的 Hugging Face 快取，它屬於它的主人。

**移除。** `cdui packs remove <pack> <item>`，或項目旁邊的刪除按鈕，會刪掉下載的檔案以及由它衍生出來的東西（轉好的 GloVe npz 會跟著原本的表一起走），並清掉紀錄。Python 套件則是刻意不動的：把 `sentence-transformers` 從「正在執行伺服器的那個直譯器」底下抽掉，不是那台伺服器可以對自己做的事，所以指令只會把該執行的那一行印出來，交給你自己跑：

```text
uv pip uninstall --python <path-to-venv-python> sentence-transformers
```

請在伺服器停掉之後再執行它。

**透過網路安裝。** 所有會造成變更的 `/api/packs` 端點，在伺服器不是綁定在回送（loopback）位址時一律拒絕，因為啟動安裝等於對「正在服務這個請求的那一個直譯器」執行套件管理程式。刻意對區網提供服務的教室或公司環境，可以用 `CODEFYUI_ALLOW_REMOTE_PACK_INSTALL=1` 重新開放；不論開不開，能被要求的東西都只限於型錄裡的項目。詳見 [API 參考](../advanced/api-reference.md)。

## 畫布上會看到的變化

套件包不在的時候，編輯器在你按下執行之前就會先說：`TextEmbedding` 與 `HFTextGenerate` 在節點面板裡是灰的 —— 這兩個就是只能從套件包來的節點 —— 而 `WordVector` 的 **backend** 下拉選單會把缺少下載的選項變灰，`demo-16d` 則仍然可以選。變灰的選項本身就帶著安裝入口，所以在套件中心面板裡點一下就能補上。若這個版本還沒有套件中心面板，`cdui packs list` 會顯示同樣的資訊。

如果執行時走到某個節點、而它需要的東西不在，執行會停在那個節點，並給出指名道姓的那句話：

```text
Model 'all-MiniLM-L6-v2' from the Sentence embeddings pack is not downloaded. Open Package Center (toolbar > Settings > Optional packs) to download it; graph runs never download (pack=sentence-embeddings)
```

結尾的 `(pack=<id>)` 是刻意設計成機器讀得懂的：編輯器會從這句訊息裡把 id 讀回來，直接給你「補上這一個下載」的按鈕。執行本身完全不會下載任何東西，所以就算網路是按流量計費的，把圖丟著跑也是安全的。

## 套件相關節點參考

### WordVector

每個輸入的詞給一個向量，來源可以是查表，也可以是編碼器。你挑哪一個 backend，本身就是那堂課要教的東西：

| Backend | 需要 | 教的是什麼 |
|---------|------|------------|
| `demo-16d` | 不需要任何東西 | 59 個詞、16 個手工設計的維度（王室、神性、性別、動物類別、移動、交通工具、食物、天氣）。內建在程式裡；經典的類比題在它上面是「設計成剛好成立」的 |
| `glove-50d` | `word-vectors` | 真正的 400,000 詞 GloVe 詞向量表。同一個類比題在這裡只是近似成立，而這個落差正是重點 |
| `sentence-transformers/all-MiniLM-L6-v2` 以及另外三個模型 id | `sentence-embeddings`（其中那一個模型） | 現代的編碼器，一次餵一個詞進去。對單一詞來說結果更亂，因為這些模型是用句子訓練的；但它才是真實檢索系統實際在用的東西 |

`normalize` 會把每一列做 L2 正規化，這樣下游做內積就等於餘弦相似度。`keep_oov` 會為表裡沒有的詞輸出一個零向量，而不是把它丟掉，而且只對兩個查表 backend 有意義：編碼器對任何字串都給得出向量，所以根本不存在「不在詞彙表裡」這件事。

**已淘汰的 backend 名稱。** 早期預覽版存下來的圖可能還帶著 `glove-100d` 或 `minilm-sentence-384d`。這兩個都會直接丟出一個指出替代選項的錯誤，而不是給你下載按鈕 —— 因為沒有任何下載能修好一個已經不存在的名字：請把 backend 分別改成 `glove-50d` 與 `sentence-transformers/all-MiniLM-L6-v2`。

### TextEmbedding

用真正的句子編碼器，把每一段文字變成一個稠密向量，於是意思相同的兩段文字，向量會指向同一個方向。語意搜尋與 RAG 就是建立在這個節點上：文件先嵌入一次，問題也嵌入，然後比較。整個節點都需要 `sentence-embeddings`。

`texts`（一個清單，例如切塊節點的輸出）和 `text`（單一字串）接其中一個就好，不能兩個都接 —— 兩個都接的圖等於對「要嵌入什麼」講了兩件不同的事。值得知道的參數：

- **`model`** —— 要載入四個編碼器中的哪一個；見下面的表。
- **`prefix`** —— 編碼前加在每一段文字前面。`multilingual-e5-small` 訓練時，問題用 `query: `、文件用 `passage: `；另外三個模型會忽略它。
- **`split_lines`** —— 預設開啟，文字輸入的每一個非空白行各自算一段文字。當你希望一份多行的文件變成單一個向量時，把它關掉。
- **`max_seq_length`** —— 每段文字的 token 上限。`0` 表示沿用模型自己的預設（paraphrase-multilingual 是 128、all-MiniLM 是 256、bge 與 e5 是 512）。超過的部分會被截掉，所以切塊大小要照著抓。
- 另外還有 **`normalize`**（預設開啟）、**`batch_size`**、**`label_chars`** 與 **`device`**。

`embeddings` 與 `labels` 兩個輸出可以直接接到 `CosineSimilarity` 與 `EmbeddingScatter`。[範例集](./examples-gallery.md)裡的 **Sentence Similarity (zh-TW)** 就是整條鏈，套件包裝好之後就能直接執行。

### RAG 鏈

檢索增強生成就是七個節點接成一排，其中只有兩個需要下載：

```text
DocumentLoader -> TextChunker -> TextEmbedding -> VectorStore -> Retriever -> PromptBuilder -> HFTextGenerate
                                                                                           (或 LLMChat)
```

| 節點 | 做什麼 | 需要 |
|------|--------|------|
| `DocumentLoader` | 讀取指定資料夾裡的每一個 `.md` 與 `.txt`（不支援 PDF、HTML、DOCX），每份文件輸出成 `{text, source}`，讓後面每一個切塊都還說得出自己來自哪個檔案。`recursive` 會連子資料夾一起讀，`max_docs` 則限制最多讀幾個檔案；把 `source` 改成 `uploaded_file`，它就改讀 `file` 選到的那一個 `.txt`，也就是用旁邊那個按鈕上傳的檔案 | 不需要 |
| `TextChunker` | 把每份文件切成小到可以嵌入、也塞得進提示詞的片段。`characters` 是固定長度的視窗，也是最不挑語言的一個 —— 中文根本沒有空格；`sentences` 與 `paragraphs` 則按作者本來就寫下的界線切，再打包到 `chunk_size` 為止。每個切塊都帶著來源與 `start_char`/`end_char`，而且 `text[start_char:end_char]` 就正好是那個切塊，所以引用是可以被驗證的，不只是裝飾 | 不需要 |
| `TextEmbedding` | 每個切塊一個向量，問題也是一個。兩邊必須指定同一個模型 | `sentence-embeddings` |
| `VectorStore` | 把切塊的向量疊成一個 `[N, D]` 矩陣，並把切塊的文字與 metadata 放在旁邊。這就是 RAG 系統裡的那個「資料庫」；每一列都以單位長度儲存，所以一次餘弦搜尋就只是一次矩陣相乘。它只存在記憶體裡，重跑時用快取好的嵌入在幾毫秒內重建 | 不需要 |
| `Retriever` | 拿問題和每一列算分數，留下最接近的 `top_k` 個，把低於 `min_score` 的丟掉，再把切塊的文字連同它們各自來自哪個檔案交給下一棒。它的紀錄行會印出每一個命中的分數 | 不需要 |
| `PromptBuilder` | 把檢索到的切塊和問題貼進一個範本，而這個範本會要求模型「只」根據那些內容回答。範本必須包含 `{context}` 與 `{question}`；想自己寫的話，接一個 `TextInput` 到 `template` 輸入 | 不需要 |
| `HFTextGenerate` | 由 Qwen2.5-0.5B-Instruct 在本機作答，chat template 會自動幫你套用，token 也會邊產生邊串流到畫布上 | `rag` |

最後一棒也可以換成 `LLMChat`：同一份提示詞，改送給本機的 Ollama 或某個雲端供應商，而不是載進這個行程裡，而且它完全不需要任何套件包。

**隨附的語料。** `backend/data/samples/rag` 附了五份短筆記 —— CodefyUI 是什麼、節點與連線、訓練基礎、嵌入與 RAG，以及選用套件包 —— 每一份都有英文和繁體中文兩半，所以一按執行就有東西可讀，多語言編碼器也才有東西可以證明自己。`DocumentLoader` 會把它被指到的那個資料夾裡的每一個 `.md` 與 `.txt` 都讀進來，所以要把這堂課換成你自己的教材，最快的方式就是把 `directory` 指向你自己的筆記資料夾；圖裡其他東西都不用改。

**e5 的前綴不是裝飾。** `multilingual-e5-small` 訓練時兩邊是不對稱的：提問那一側加 `query: `，建索引那一側加 `passage: `。所以兩個 `TextEmbedding` 節點的前綴不同、`model` 卻必須相同 —— 而「同一個模型」是這張圖唯一不能被破壞的前提。兩個模型就是兩個向量空間，跨空間算出來的餘弦值只是一個沒有意義的數字：只改其中一側的編碼器，搜尋照樣會回傳 `top_k` 個切塊，只是那些切塊是隨便挑的，而且圖裡沒有任何東西擋得下來。

**「只根據上下文回答」才是整堂課的重點。** 一個 0.5B 的模型對 CodefyUI 幾乎一無所知。把五份關於它的筆記擺在模型面前，它卻能答對 —— 不是因為它被微調過，而是因為對的段落被貼在問題前面了。這個落差就是檢索存在的理由，而範本則負責讓它誠實：問一個語料裡根本沒有答案的問題，`Retriever` 還是會回傳它最近的那幾個切塊，因為「最近」永遠都定義得出來；真正讓模型說「我不知道」而不是就地編一個的，是那一句指示。

**在 CPU 上生成是什麼感覺。** 筆電上大約每秒幾個 token，所以一個答案大概要等幾秒到幾十秒 —— 本機那個範例會在模型結束這一輪時就停下來，以內附的那個問題來說，遠不到它 160 個 token 的上限 —— 第一次執行還要多花幾秒把權重從磁碟讀進來。這些都是依模型大小估出來的數字，不是碼表量出來的。節點會逐個 token 回報進度，所以你看得出它正在工作，而不是只是卡著。有 GPU 會快非常多；`device` 預設跟著全域裝置選擇器走，除非你在節點上另外指定。

兩張圖都在[範例集](./examples-gallery.md)裡，而且每一張旁邊都有一份 `README.md`。**RAG, fully local**（`examples/LLM/RAG-Local-Offline`）需要兩個下載 —— `qwen2.5-0.5b-instruct` 與 `multilingual-e5-small` —— 之後執行時不會有任何東西離開這台機器。**RAG with a chat API**（`examples/LLM/RAG-LLMChat-API`）的檢索鏈和它一個節點一個節點完全相同，只是最後一棒換成 `LLMChat`，所以它只需要編碼器，外加一個跑著的 Ollama 或某個供應商的金鑰。拿同一個問題把兩張圖都跑一次，正是最值得做的對照：檢索到的上下文一模一樣，所以答案上的每一個差異都來自生成模型。

## 如何挑選嵌入模型

| 模型 | 語言 | 維度 | token 上限 | 需要前綴嗎 | 下載量 |
|------|------|-----:|-----------:|------------|-------:|
| `sentence-transformers/all-MiniLM-L6-v2` | 英文 | 384 | 256 | 不需要 | 90 MB |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`（預設） | 50 種以上，含繁體中文 | 384 | 128 | 不需要 | 470 MB |
| `BAAI/bge-small-zh-v1.5` | 中文 | 512 | 512 | 不需要 | 95 MB |
| `intfloat/multilingual-e5-small` | 100 種以上，含繁體中文 | 384 | 512 | `query: ` / `passage: ` | 470 MB |

- **只用英文、而且想要最小的下載** —— `all-MiniLM-L6-v2`。它同時也是四個裡面最快的。
- **繁體中文，或是一堂多語言混用的課** —— 用預設那一個。它不需要任何前綴，課堂上就少一件會弄錯的事，而且它對齊語言的能力足以讓一句中文和它的英文翻譯落在相鄰的位置。
- **只有中文、而且段落比較長** —— `bge-small-zh-v1.5`：每段 512 個 token（預設那個只有 128），只要 95 MB。
- **做檢索，而且問題和文件是分開嵌入的** —— `multilingual-e5-small`，問題加 `query: `、文件加 `passage: `。不加前綴的話，它的分數會比其他三個還差，而且不會報任何錯 —— 這是最安靜的那種錯。

同一時間最多有兩個模型留在記憶體裡，這剛好就是「拿英文模型和多語言模型互相比較」所需要的量；載入第三個時，最久沒用到的那一個會被丟掉。

## 疑難排解

- **「reports installed but `sentence_transformers` cannot be imported」** —— 紀錄檔說套件包裝好了，直譯器卻不同意，這是安裝壞掉，而不是少下載。請從套件中心重新安裝，或執行 `cdui packs install sentence-embeddings --yes`。
- **「Model ... is not downloaded」** —— Python 那一半在，但那一個模型不在。四個模型是互相替代的，裝了一個絕不會順便裝其他的：請在套件中心下載它，或執行 `cdui packs install sentence-embeddings --items multilingual-e5-small`。
- **CPU 上的速度。** 這些都是小模型（22M 到 118M 參數），CPU 本來就是它們預期要跑的地方。一次工作階段裡的第一次編碼要花幾秒把權重從磁碟讀進來，之後幾句話的編碼遠低於一秒。GloVe 則要付一次性的轉檔成本：把下載回來的文字表轉成 npz，大約十秒，而且過程中會有一行進度說明；之後每個行程載入它大約一秒。
- **生成很慢。** `HFTextGenerate` 是一個 token 一個 token 解碼的，0.5B 的模型在筆電 CPU 上一秒只能吐出幾個，所以一個答案要等上幾十秒，這裡沒有什麼壞掉的東西要修。可以動的地方，由便宜到貴：把 `max_new_tokens` 調小（它是「這個節點最久會跑多久」的上限，不是目標值）；把 `Retriever` 的 `top_k` 調小，或用 `PromptBuilder` 的 `max_context_chars` 設個上限，因為每一個檢索回來的字元都要先被讀過，模型才會寫出第一個 token；有 GPU 的話把 `device` 設成 `cuda`。把最後一棒換成 `LLMChat`，則是直接把生成整個移出這個行程。
- **答案沒有理會上下文。** 先看 `Retriever` 的紀錄行再怪模型 —— 它會印出每一個命中的分數。最高分只有 0.3 左右，通常表示語料裡根本沒有這個答案，換什麼提示詞都救不回來。如果命中看起來是對的、答案卻在亂飄，就把 `top_k` 調大，好讓真正回答問題的那一段確實進到提示詞裡。如果什麼都沒回來，那是 `min_score` 濾掉的：這時 `PromptBuilder` 會寫進 `(no context retrieved)` 並發出警告，把門檻調低（0 表示全部保留）就會把切塊放回來。如果連分數本身都像是隨便給的，請檢查兩個 `TextEmbedding` 節點是不是指定了同一個模型 —— 這個錯誤產生的是聽起來很合理的胡說，而不是一個錯誤訊息。
- **Windows 的路徑。** Hugging Face 的 snapshot 目錄層數很深。如果快取本來就位在一條很長的路徑底下，請開啟長路徑支援，或用 `CODEFYUI_USER_DATA_DIR` 把它指到淺一點的地方。另外在 Windows 上，移除有可能回報「紀錄已清除，但檔案還在磁碟上」，因為有程式正開著它們：請停掉伺服器，再手動刪掉那個目錄。
- **「cannot be installed while the server is running」。** 每一次線上安裝都跑在一份 constraints 檔底下，它把這個直譯器裡已經有的每一個發行套件都釘在目前的版本，所以安裝只能「增加」—— 執行中的伺服器已經載入的東西，不可能在它腳下被換掉。若某個套件包非得換掉某個東西，它會直接停下來（用 CLI 時的離開碼是 3）而不是換到一半，並印出一行要在伺服器停掉之後執行的 `uv pip install` 指令。請用 `cdui stop` 停掉伺服器、執行那一行，再重新啟動。GPU 套件包永遠屬於這一類：它是用 `cdui install --gpu <variant>` 安裝的，不是 `cdui packs install`。
- **磁碟空間不夠。** 在抓第一個位元組之前就會先檢查，所以不會發生 470 MB 下載到 90% 才發現這顆磁碟從一開始就放不下。訊息會告訴你需要多少、目前剩多少。

## 授權

型錄裡的每一項都採用寬鬆授權，而且授權資訊跟著項目走：`cdui packs list` 會把它印在每一個下載旁邊，而 `backend/app/core/packs/catalog.py` 就是它被寫下來的地方。

| 項目 | 授權 |
|------|------|
| `sentence-transformers`（Python 套件） | Apache-2.0 |
| `all-MiniLM-L6-v2` | Apache-2.0 |
| `paraphrase-multilingual-MiniLM-L12-v2` | Apache-2.0 |
| `bge-small-zh-v1.5` | MIT |
| `multilingual-e5-small` | MIT |
| `glove-50d`（glove-wiki-gigaword-50） | PDDL-1.0 |
| `qwen2.5-0.5b-instruct` | Apache-2.0 |

CodefyUI 本身採用 AGPL-3.0，另有商業授權選項，見 [授權](../licensing.md)。套件包的內容維持它們各自的授權，而且是從它們自己的上游下載的，所以這個專案並沒有轉散布其中任何一項。
