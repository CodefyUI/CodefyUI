---
sidebar_position: 3.95
title: 建立自己的評估流程
description: 用 eval 外掛在自己的資料上評估 Hugging Face 模型：含 id 與雜湊的評估集、鎖定版本的模型、預測表、指標，以及依項目對齊的 A/B 比較。
---

# 建立自己的評估流程

`eval` 外掛把模型評估變成一張圖：讀取評估集、以一個確切的 revision 載入模型、逐筆跑完每個項目、為答案評分，並逐筆比較兩次執行。圖中印出的每個數字，都能追溯到產生它的權重、tokenizer、設定與檔案。

本頁用你自己的資料，一次一個節點地建立這張圖。外掛內附的兩個範例圖 **A/B text classifier eval** 與 **VQA eval skeleton** 就是本頁內容的完成版；從範例庫開啟其中一個即可對照。逐節點的完整說明（包含預測表的每一欄）請見外掛的 [README](https://github.com/CodefyUI/CodefyUI/blob/main/plugins/eval/README.md)。

## 開始之前 {/* #before-you-start */}

從[外掛中心](./plugin-center.md)安裝此外掛，或在終端機執行：

```bash
cdui plugin install eval
cdui plugin install stats   # 選用：顯示結果的圖表與表格
```

安裝 `eval` 時也會安裝 `transformers`，這是 Hugging Face 載入節點唯一需要的套件。模型在第一次使用時下載到 CodefyUI 自己位於使用者資料夾下的快取。以 `HF_HUB_OFFLINE=1` 啟動伺服器時只使用已快取的內容：不會下載任何東西，未快取的模型會直接報錯並說明原因。

## 1. 撰寫評估集 {/* #write-the-eval-set */}

評估集是一個 JSONL 檔：每行一個 JSON 物件，每個項目一行。

```json
{"id": "s01", "text": "A warm, funny film with a big heart.", "label": "positive"}
{"id": "s02", "text": "Two hours I will not get back.", "label": "negative"}
```

加入一個 **Eval-JSONLDataset** 節點，把 `path` 設為該檔案：可以是上傳檔案的名稱、相對於後端工作目錄（專案模式下為專案目錄）的路徑，或用 `eval:samples/...` 指向外掛內附的檔案。以 `input_field`（模型看到的文字）、`label_field`（預期答案；每列可有可無）與 `id_field` 指定欄位。

- **id 是項目的識別。** 兩次執行依 id 比較，絕不依列的位置，因此重複的 id 在載入時就會被拒絕。沒有 id 的檔案會得到 `line-N`，行的位置一變它就跟著變；請為項目加上 id。
- **檔案也有識別。** 資料集記錄路徑、檔案位元組的 sha256 與列數，由它產生的每張預測表都帶著這些資訊。
- **影像。** 做視覺問答時，在每筆紀錄加上影像路徑，並把 `image_field` 設為該欄位。路徑相對於 JSONL 檔所在的資料夾，不可包含 `..`，也不可是絕對路徑，讓資料集資料夾自成一體。每張影像都在載入時讀取並計算雜湊；缺少的影像會被拒絕並指出行號，資料集也會多出 `images_sha256`，所以檔名不變但內容改變的影像，也會改變資料集的識別。

```json
{"id": "q01", "image": "shapes-vqa/red-ball.png", "question": "What color is the ball?", "answer": "red"}
```

## 2. 套用提示範本（選用） {/* #template-the-prompt */}

**Eval-PromptTemplate** 改寫每個項目的輸入，例如為因果語言模型寫成 `Review: {input}\nSentiment:`。`{input}` 是目前的輸入，`{欄位}` 是紀錄中的任一欄位。套用後的文字就是模型看到、也是輸入雜湊記錄的內容；範本的文字與雜湊會加入資料集的識別。用到標籤欄位的範本會被拒絕，因為那等於把答案給模型看。

## 3. 以固定 revision 載入模型 {/* #load-the-model */}

**Eval-HFModelLoader** 接受 Hugging Face Hub 的 id（`org/name`）或本機模型目錄，以及一個 `revision`。revision 會在載入前解析成 commit sha，模型從那個確切的快照載入。要讓一張圖的數字固定不變，請把 `revision` 設為 commit sha：模型擁有者每次推送，`main` 都會移動。本機目錄以其中模型與 tokenizer 檔案的 sha256 作為識別。

`task` 決定模型的用法。`auto` 會讀取模型的設定：

| `task` | 模型 | 答案 |
|---|---|---|
| `classification` | `...ForSequenceClassification` | 最可能的標籤，以模型的 `id2label` 命名 |
| `generation` | 因果語言模型（`...ForCausalLM`、`...LMHeadModel`） | 接在提示之後解碼出的文字 |
| `vqa` | `ViltForQuestionAnswering` | 針對一張影像與一個問題，從模型的答案詞表中選出最可能的答案 |

把模型接到 **Eval-HFProcessorLoader**，並把它的 `model_id` 與 `revision` 留空：tokenizer 就會從模型的 repository、在同一個 commit 載入。來自同一個 repository 其他 commit 的 processor 會被拒絕。`vqa` 模型請把 `kind` 設為 `auto`，影像 processor 才會與 tokenizer 一起載入；只有 tokenizer 會被拒絕。

## 4. 分批推論 {/* #run-batched-inference */}

**Eval-BatchInference** 接收模型、processor 與資料集，產生一張**預測表**：每個項目一列，含 `id`、`input`、`input_hash`、`raw_answer`、正規化後的 `answer`、`label`、`status` 與 `error`。分類與 VQA 的列另有 `score` 與 `topk`；VQA 的列還會記錄 `image` 與 `image_sha256`，其 `input_hash` 同時涵蓋問題與影像。

- `normalize`（預設 `strip_lower`）在任何比較之前套用到答案與標籤。`label_map` 先改名原始答案，用於設定中沒有標籤名稱的模型：`LABEL_0=negative, LABEL_1=positive`。
- **狀態。** 已完成批次的項目是 `ok`，之後也維持 `ok`。出錯的批次會逐筆重試，所以一個有問題的輸入（例如無法讀取的影像）只會產生一列 `error`。按下停止後，執行會在批次之間結束：尚未開始的項目標成 `cancelled`，表格仍包含每個項目，指標會略過未完成的列並說明略過了幾列。
- **生成設定。** 預設以貪婪解碼生成，並使用此節點的參數（`do_sample`、`temperature`、`top_p`、`repetition_penalty`、`seed`）。把 `generation_config` 設為 `model`，就會改從模型自己的 `generation_config.json` 取得 `do_sample`、`temperature`、`top_p`、`top_k` 與 `repetition_penalty`，也就是 `model.generate` 會使用的設定。不論哪一種，表格都會記錄實際使用的設定，並在旁邊記錄模型建議的設定。抽樣使用的亂數產生器以 `seed` 與項目的 id 為每個項目各自設定種子，所以一個項目的答案不受批次大小或項目順序影響。

## 5. 評分 {/* #score-it */}

把預測表接到一個或多個指標節點。每個指標只計算狀態為 `ok` 且有標籤的列，在數值旁回報 `counted` 與 `skipped`，並以 `metric_name` 把數值記錄成 run metric。

| 節點 | 數值 | 適用 | 對照的參考實作 |
|---|---|---|---|
| **Eval-ExactMatch** | 答案等於標籤的比例 | 所有任務 | `sklearn.metrics.accuracy_score` |
| **Eval-F1Score** | micro 或 macro F1 | classification、VQA | `sklearn.metrics.f1_score` |
| **Eval-TopKAccuracy** | 標籤落在前 `k` 個答案內的比例 | classification、VQA | `sklearn.metrics.top_k_accuracy_score` |
| **Eval-BLEU** | 語料層級 BLEU，0 到 100 | generation | `sacrebleu.corpus_bleu` |
| **Eval-ROUGE-L** | 平均 ROUGE-L F-measure（或 precision、recall），0 到 1 | generation | `rouge_score` 的 `RougeScorer(["rougeL"])` |

BLEU 與 ROUGE-L 由外掛自行計算，因此不必為它們額外安裝套件；外掛的測試會把每個數值與參考實作比對。處理非英文文字時有兩個設定要注意：

- **Eval-BLEU** 的 `tokenize` 預設為 sacrebleu 的 `13a`；中文請用 `zh`（逐字切開中文字）或 `char`。metrics 表格會記錄 sacrebleu 的 signature，例如 `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp`。
- **Eval-ROUGE-L** 的 `tokenize` 為 `rouge` 時使用 rouge-score 的 tokenizer，只保留 `a-z` 與 `0-9`：中文答案沒有任何 token，得分為 0。這類文字請用 `char`。

為每次執行的指標取不同的 `metric_name`（`exact_match_a`、`exact_match_b`），參數掃描與執行紀錄才分得出來。

## 6. 比較兩次執行 {/* #compare-two-runs */}

用另一個模型，或同一個模型的另一個 revision，再建立一次同樣的節點鏈，然後把兩張預測表都接到 **Eval-CompareRuns**。各列依 `id` 對齊；遇到下列情況時拒絕比較，並列出最先出問題的項目：

- 任一張表中有重複的 id；
- 兩邊的 id 集合不同；
- 同一個 id 的 `input_hash` 不同（要刻意比較兩種提示時，請關閉 `require_same_input`）；
- 同一個 id 的標籤不同。

差異表列出每個項目的兩邊答案、是否一致，以及哪一邊正確；某一邊沒有完成的項目會連同其狀態留在差異表中。`summary_table`、`summary_columns` 與 `summary_rows` 遵循 stats 外掛的表格格式，可以直接接到 **Stats-ChartView** 與 **Stats-TableView**；`chart` 是一致性矩陣。

## 解讀來源資訊 {/* #reading-the-provenance */}

預測表的 `meta` 記錄讀者信任或重現結果所需的資訊：模型與 processor 的 id、要求的與解析出的 revision、實際生效的推論與生成設定、seed，以及資料集的路徑、sha256、列數、範本與（有影像時的）`images_sha256`。只有在兩邊的 revision 都鎖定到 commit 或內容雜湊、資料集有雜湊、且抽樣生成有 seed 時，`meta.reproducibility` 才為 true；否則 `reasons` 會列出缺少的項目。只有模型 id 永遠不算可重現。

## 內附的兩個範例 {/* #the-shipped-examples */}

| 範例 | 內容 | 結果 |
|---|---|---|
| **A/B text classifier eval** | 兩個很小的 BERT 情感模型跑同樣 20 句，逐筆比較 | exact match 1.0000 與 0.9500；只有 `s15` 不一致 |
| **VQA eval skeleton** | ViLT（`dandelin/vilt-b32-finetuned-vqa`）回答關於 7 張手繪圖片的 12 個問題 | exact match 0.5000，top-3 accuracy 0.8333 |

兩個範例都把模型鎖定在一個 commit。第一次執行時，每個情感模型約下載 18 MB，ViLT 約 470 MB。ViLT 是從照片學習的，範例中的平面圖形和照片差異很大，所以 VQA 範例的分數反映的是流程如何呈現一個模型，對 ViLT 本身的能力說明有限。

## 從 VQA 到影片語言模型評估 {/* #from-vqa-to-video */}

VQA 範例是影片語言模型評估的骨架：輸入視覺內容與一個問題，輸出一個答案並與標籤比對評分，視覺內容的雜湊會加入每個項目的識別。要評估自己的模型，請把 JSONL 與影像換成自己的資料集，並把載入節點的 `model_id` 與 `revision` 換成自己的 checkpoint。

`vqa` 任務適用於從固定詞表中選出答案的模型，ViLT 就是這一類。以文字寫出答案的生成式視覺語言模型，以及影片輸入（從片段取樣的畫格），目前還不是此外掛支援的任務。

## 檢查與限制 {/* #checks-and-limits */}

- 範例圖與指標都能在 CPU 上執行。大型模型可透過載入節點的 `device` 在 GPU 上執行。
- HF 載入節點會在記憶體中保留最近載入的兩個模型，供 A/B 比較使用。
- `Eval-BatchInference` 不會使用快取：它的模型輸入是即時物件，所以每次執行都會重新評估。
- 測試涵蓋的範圍，以及如何執行選用的 Hub 檢查，列在 README 的 Tests 一節。
