import type { ExampleTranslations } from './types';

/**
 * Traditional Chinese for the example gallery, keyed by the `path`
 * `/api/examples/list` returns.
 *
 * Names are absent on purpose: an example's English name is its handle in the
 * docs, in `examples/` on disk and in a `run_graph.py` argument, so it reads
 * the same in both locales and only the description changes.
 *
 * Each description is written to survive the card's 80-column cut: the first
 * sentence says what the graph shows, and any requirement -- a GPU, a
 * download, a pack that has to be installed first -- sits inside that cut
 * rather than after it. `exampleLocales/zh-TW.test.ts` pins both.
 */
const zhTW: ExampleTranslations = {
  // ── Usage examples -- the beginner-facing starters ──
  'Usage_Example/Api-Function': {
    description:
      '把一張圖當成函式呼叫的示範：Start → GraphInput → Print → GraphOutput。在畫布上直接執行會用輸入的預設值；存檔之後也可以用 POST /api/graph/run/Api-Function 帶 {"inputs": {"message": "..."}} 呼叫，拿回 {"outputs": {"echo": "..."}}。詳見文件「使用方式 → 把 graph 當成函式呼叫」。',
  },
  'Usage_Example/CNN-MNIST/InferenceCNN-MNIST': {
    description:
      '必須先跑過一次「Train CNN on MNIST」：那張圖會寫出 model_weights.pt，這張圖才載得到。安裝檔不會附訓練好的權重，所以在全新的機器上，沒訓練過就會停在 ModelLoader。之後 ImageReader 讀進 test_digit.png（backend/data/images/ 裡附的真實 MNIST 數字），SequentialModel 重建同一套架構，ModelLoader 還原權重，Inference 做一次前向傳播，辨識這張手寫數字。',
  },
  'Usage_Example/CNN-MNIST/TrainCNN-MNIST': {
    description:
      'SequentialModel 以 JSON 層設定組出 nn.Module，接上 Training Pipeline 預設模組（Dataset → DataLoader → Optimizer → Loss → TrainingLoop）完整訓練。結束後 ModelSaver 把權重寫到 backend/data/models/model_weights.pt，推論範例才載得到。',
  },
  'Usage_Example/GPT-Mini/TrainGPT-Mini': {
    description:
      'SequentialModel 把每張 28x28 影像攤平，用一個學出來的 embedding 投影成 16 token × 24 維的「序列」，交給 4 層 TransformerDecoder（自注意力模式）處理，最後投影成 10 個類別 logits；訓練走 Training Pipeline 預設模組。打開 SequentialModel 的層編輯器就能改 d_model、層數或 head 數。',
  },
  'Usage_Example/ResNet-CIFAR10/TrainResNet-CIFAR10': {
    description:
      'SequentialModel 組出一個含兩個殘差區塊的 mini-ResNet（捷徑邊 stem_pool→b1_add 與 b1_r2→b2_add）並輸出成 nn.Module，再接上 Training Pipeline 預設模組（Dataset → DataLoader → Optimizer → Loss → TrainingLoop）完整訓練。想看或改殘差結構，打開 SequentialModel 的層編輯器。',
  },
  'Usage_Example/ResNet18-CIFAR10-Baseline': {
    description:
      '完全用 GUI 節點搭出來的研究等級 CIFAR-10 baseline。架構是 CIFAR 版的 ResNet-18（3x3 stride-1 stem、不接 maxpool、4 個 stage 各 2 個 BasicBlock，11,173,962 個參數），寫在 SequentialModel 的層編輯器裡。配方：SGD（Nesterov）lr 0.1、momentum 0.9、weight decay 5e-4、batch 128、整段排程走 cosine annealing，增強用 RandomCrop(32, padding=4) + RandomHorizontalFlip + CIFAR-10 通道正規化。從執行任務面板以固定種子送出執行；實測準確率見範例目錄下的 README。',
  },

  // ── Classical ML ──
  'Classical/Iris-Sklearn-KNN': {
    description:
      '正式版的 KNN 分類器（封裝 sklearn）：距離加權投票、內部建 KD-tree 索引，所以同一張圖在 10 萬列以上的資料仍然跑得動。這是雙軌配對的另一半 — 教學版是 EduKNN 節點，每一段距離都攤開來算給你看。兩者的輸入/輸出介面相同，要互換只需要改一個節點。EduKNN 與對應的 KNN-from-Scratch 範例在 foundations 章節包裡（cdui plugin install foundations），基本安裝不含。',
  },
  'Classical/Tabular-Iris-Pipeline': {
    description:
      '載入 Iris（150 列 × 4 特徵 × 3 個品種），四個特徵全選，逐欄做 z-score 正規化，再以 80/20 分層切分，讓訓練集與測試集都看得到三個品種 — C1-2 課程的具體版本。最後的 Print 印出訓練特徵張量，可以核對形狀 (120, 4)，並確認各欄平均值都落在 0 附近。',
  },

  // ── LLM -- embeddings, retrieval, causal-LM training ──
  'LLM/RAG-LLMChat-API': {
    description:
      '需要 sentence-embeddings 套件包，外加一個跑著的 Ollama，或改用 API 金鑰。檢索留在本機，生成換成 LLMChat：預設連本機 http://127.0.0.1:11434 上的 qwen2.5:0.5b（先 `ollama pull qwen2.5:0.5b`）；把 provider 改成 ChatGPT API 或 Claude API 並給金鑰，同一份提示詞就會送到雲端模型 — 內容會離開這台機器。檢索鏈與 RAG-Local-Offline 完全相同（DocumentLoader → TextChunker → TextEmbedding → VectorStore → Retriever → PromptBuilder），只換掉最後的生成節點，答案可以跟本機版對照。各階段說明在旁邊的 README.md。',
  },
  'LLM/RAG-Local-Offline': {
    description:
      '需要 rag 與 sentence-embeddings 兩個套件包（約 1.5 GB），裝好之後就不必連網。DocumentLoader 從 data/samples/rag 讀五篇中英雙語短文，TextChunker 切成 400 字元的塊，TextEmbedding（multilingual-e5-small，前綴 `passage: `）轉成向量，VectorStore 建索引；問題以同樣方式嵌入（前綴 `query: `），Retriever 撈最接近的 3 塊，PromptBuilder 包成「只依內容作答」的提示詞，HFTextGenerate 再用 Qwen2.5-0.5B-Instruct 寫出答案。CPU 上生成要數秒到數十秒，內附的問題大約 30 個 token 就結束。',
  },
  'LLM/Sentence-Similarity-zhTW': {
    description:
      '多語言 sentence-transformer 編碼八個句子。需要 sentence-embeddings 套件包（套件中心），裝好後在 CPU 上離線執行。四組配對分別是天氣、食物、股市、機器學習，最後一組是中文對英文；CosineSimilarity 為每句列出最近的 3 句 — 第 1 名永遠是自己（1.0），第 2 名應該是它的配對句，即使兩句沒幾個字相同，中英那組則顯示模型把兩種語言對齊。EmbeddingScatter 把向量投影到 2D；想要中文專用的編碼器，換成 BAAI/bge-small-zh-v1.5。',
  },
  'LLM/TrainCausalLM-TinyStories': {
    description:
      '需要 16 GB 顯卡，第一次執行會下載一次語料：在 TinyStories 上從零預訓練一個 204M 參數的 GPT 風格 decoder，訓練完算 perplexity，並讀它寫出來的故事。一個 epoch 是 2,441 個 micro-batch（bf16）、約 610 次優化器更新 — 語料有設上限，所以一小時內跑得完。完整配方、token 預算與記憶體調節手段在旁邊的 README.md。',
  },
  'LLM/Word-Embedding-Analogy': {
    description:
      '取 king、man、woman 三條詞向量，用兩個 Add 節點組出類比（alpha=-1 就是減掉 man），再讓 CosineSimilarity 從列出的 59 個字裡找最接近的一個；排除 king、man、woman 之後答案是 queen。預設的 demo-16d 玩具詞彙表隨安裝附帶、離線可跑，類比在它上面精確成立；改成 backend=glove-50d（word-vectors 套件包，40 萬個真實單字）這題 queen 仍然勝出，但其他類比就只是近似 — 那才是重點。EmbeddingScatter 把整個詞彙表投影到 2D。',
  },

  // ── Diffusion ──
  'Diffusion/Forward-Process': {
    description:
      '一個 Lerp（alpha=0.4）就是前向加噪公式的教學替身：結果是乾淨訊號與高斯噪聲的 40/60 混合，固定的 alpha 取代了整條加噪排程。完整式子為 $x_t = \\sqrt{\\bar\\alpha_t}\\,x_0 + \\sqrt{1-\\bar\\alpha_t}\\,\\epsilon$，t 越大，$\\bar\\alpha_t$ 就從 1（乾淨）降到 0（純噪聲）。旁邊的 TimestepEmbedding 示範純量 t 如何變成 32 維向量，讓 U-Net 拿它當條件訊號。',
  },
  'Diffusion/Mini-UNet-Compact': {
    description:
      '整個 U-Net 收在一個 `DiffusionUNet` 節點裡，不必手接每一顆 ResBlock、Upsample 與 Concat — 真要組取樣流程時就是用這個形式。跑跑看：GaussianNoise 餵給模型，DDPMSampler 只走一步，確認輸出形狀不變。想看同一套架構一塊塊攤開，對應的 `Mini-UNet-Expanded` 在 `deep` 章節外掛裡（`cdui plugin install deep`），基本安裝不含。',
  },
  'Diffusion/Toy-Sampling': {
    description:
      '反向 diffusion 的取樣迴圈，用的是剛初始化、沒訓練過的 `DiffusionUNet`。所以最後那張「影像」只是無意義的噪聲 — 但軌跡與排程的算術跟訓練好的 Stable Diffusion 完全一樣。看 DDPMSampler：20 個反向步，每一步帶不同的 timestep 呼叫 U-Net 一次。要有意義的輸出得先訓練 U-Net（另一張圖的事）；這個範例教的是取樣的數學，不是產生好看的圖。',
  },

  // ── Transformer ──
  'Transformer/MoE-TopK-Routing': {
    description:
      '一個 2 × 5 × 32 的 token 張量（2 批、每批 5 個 token、隱藏維度 32）進 MoELayer，專家數 N=4、top_k=2。每個 token 由一個小 gating 網路用 softmax 挑出最適合的 2 位專家，層的輸出是這 2 位專家 FFN 輸出的加權和。三個輸出要合起來讀：`expert_indices` 說每個 token 挑了哪 2 位、`routing_weights` 說 gate 對這個選擇有多篤定、`output` 是兩者混合的結果 — [B, T, H]，和輸入同形狀，所以 MoE 層可以直接塞進 Transformer block 裡原本放一般 FFN 的位置。Switch Transformer、Mixtral、DeepSeek-MoE 是同一套結構，這裡縮小到路由決策一個畫面就看得完。',
  },

  // ── RNN ──
  'RNN/RNN-OneStep': {
    description:
      '三個輸入向量 x_1、x_2、x_3 串過三顆 RNNCell，每顆的 hidden 輸出接到下一顆的 hidden 輸入。三顆的參數完全相同（input_size=4、hidden_size=8、seed=42），等於共用同一組 W_ih 與 W_hh — 這就是「遞迴」的全部意思。最後的 h_3，就是這個最簡架構裡「把整段序列讀完」長什麼樣子。',
  },

  // ── Reinforcement learning ──
  'RL/RLHF-Reward-and-KL': {
    description:
      '上半 RewardModel 給每條序列打一個分數，下半 KLDivergence 量策略與參考模型的距離 — 預訓練 LLM 變成 RLHF 模型靠的就是這兩塊。上半的輸入是一批假的 hidden state（4 條序列 × 16 維），實務上這顆獎勵模型要用人類偏好資料訓練。下半的「策略」是隨機 logit 張量，「參考」是全零 logit（均勻分布），KL 項的作用是不讓 PPO 離參考模型太遠。兩塊在這裡各自獨立，方便分開檢視；真的跑 RLHF 時，KL × β 會在 PPO 內部從獎勵裡扣掉。',
  },

  // ── Vision-language-action ──
  'VLA/TrainVLA-PushWorld': {
    description:
      '行為複製一個 3.3M 參數的視覺-語言-動作策略，需要 GPU、約一小時。訓練資料是 2,400 條加了 DART 噪聲的腳本示範，訓練完直接閉環評估並輸出 rollout 影片（execute_k=2 實測成功率 0.97）。把 VLARollout.instruction_mode 改成 swapped，成功率會塌到 0.03 — 策略確實在讀指令。完整配方與消融手冊在同一層的 README.md。',
  },

  // ── Model architectures -- illustrative forward passes ──
  'Model_Architecture/BERT-Encoder-Transformer': {
    description:
      '前向傳播示範：每個 token 都能注意到序列裡其他所有 token — 這就是雙向編碼器和 GPT（decoder、因果自注意力）最大的差別。輸入是 (seq=16, batch=2, d=48) 的 token 嵌入張量，走完 TransformerEncoder 堆疊後沿序列維度取 Mean 池化，再投影成 2 個分類 logits，相當於 CLS head 分類器。',
  },
  'Model_Architecture/BiGRU-SpeechRecognition-RNN': {
    description:
      '前向傳播示範：雙向 GRU 聲學模型，DeepSpeech 這類 CTC 語音辨識器的標準前端。模擬的 mel 頻譜批次 (batch=2, seq=32, features=40)，也就是 32 個時間幀、40 個 mel bin，進入 2 層雙向 GRU；因為 bidirectional=true，輸出的隱藏維度加倍成 128。逐幀的 Linear head 在每個時間步投影成 50 個音素類別。',
  },
  'Model_Architecture/ConvNeXt-CNN': {
    description:
      'ConvNeXt（FAIR, 2022）風格區塊的前向傳播。它用四項取自 Transformer 的改動把一般的 ResNet 現代化：(1) patchify stem（Conv 4x4 stride 4）、(2) depthwise 式的空間混合、(3) inverted bottleneck 的 1x1 卷積、(4) GELU 激活。這裡沒有 depthwise 的基本節點，改用一般 Conv2d，架構相近但不完全等價。(2,3,32,32) → stem（8x8 特徵圖）→ 一個帶殘差的 ConvNeXt 區塊 → pool → Linear → 10 類 logits。',
  },
  'Model_Architecture/DQN-Atari-RL': {
    description:
      'DeepMind 2015 年在 Atari 上超越人類的那個架構。Conv 層從堆疊的 84x84 畫面抽出視覺特徵，交給 DQN head 估每個動作的價值。這張圖完全離線跑：TensorCreate 產生一批合成的預處理觀測（randn，shape 2x4x84x84）取代真的 Atari 環境，不必安裝 gym/ale-py；要接真環境就換成 EnvWrapper。旁邊的 `seq-model` 不在資料路徑上，它把畫布上一顆顆接出來的 Conv/ReLU 骨幹打包成同一份層堆疊 — 雙擊可以在「模型架構」編輯器裡讀完整架構。優化器吃的是 `dqn-1.model`，也就是真正被訓練的那個 agent 網路。',
  },
  'Model_Architecture/DiT-Diffusion-Transformer': {
    description:
      '前向傳播示範：Stable Diffusion 3 與 OpenAI Sora 用的骨幹。關鍵是用 Transformer 的全域感受野，取代傳統 diffusion 裡區域性的 U-Net。帶噪影像批次 (2,4,8,8) 經 stride-2 的 Conv2d 切成 16 個 64 維 token，TransformerEncoder 讓所有 patch 互相注意，最後由 Linear head 投影回每個 patch 的噪聲預測。',
  },
  'Model_Architecture/EfficientNet-CNN': {
    description:
      'EfficientNet 核心的 MBConv 區塊前向傳播（mobile inverted bottleneck）。expand → depthwise 空間混合 → squeeze-and-excite → project 這條路徑在這裡簡化成：(1) 1x1 Conv 擴張通道（32→128）、(2) 3x3 Conv 做空間混合、(3) Squeeze-and-Excite 走 AdaptiveAvgPool→Linear→Linear→sigmoid→Multiply、(4) 1x1 Conv 投影回原通道、(5) 殘差 Add。(2,32,16,16) → MBConv 區塊 → (2,32,16,16) → 全域池化 → Linear → 10 類 logits。',
  },
  'Model_Architecture/GPT-DecoderOnly-Transformer': {
    description:
      '前向傳播示範：TransformerDecoder 把自己的輸出接回 memory，只做因果自注意力、沒有 encoder。這就是 GPT、Claude、LLaMA 這類 decoder-only LLM 的核心。輸入是 (seq_len=16, batch=2, d_model=24) 的 token 嵌入張量；後面接 Permute 轉成 batch-first、Flatten、Dropout，再用 Linear head 投影成 10 個 logits。這張圖不訓練，要訓練請看 Usage_Example/GPT-Mini。',
  },
  'Model_Architecture/LLaMA-Decoder-Transformer': {
    description:
      '前向傳播示範：架構和 GPT 幾乎一樣（因果自注意力、沒有 encoder），差別在 pre-normalization。進 attention 堆疊之前先做一次 LayerNorm（論文原本用的是 RMSNorm）。輸入是 (seq=16, batch=2, d=32) 的 token 嵌入張量：先正規化，decoder 以 tensor=memory 只做自注意力，再正規化、池化，最後投影成 32 個 vocab logits。',
  },
  'Model_Architecture/PPO-Robotics-RL': {
    description:
      'Actor-Critic 配上截斷代理目標，讓策略梯度更新穩定。PPO（OpenAI, 2017）是現在多數 RL 實際部署的預設演算法 — OpenAI Five（Dota 2）、ChatGPT 的 RLHF 微調、機械手臂操作、自動駕駛都用它。這張圖完全離線跑：TensorCreate 產生一批 Humanoid 風格的合成觀測（randn，shape 2x376）取代真的 MuJoCo 環境，不必安裝 gym/mujoco；要接真環境就換成 EnvWrapper。旁邊的 `seq-model` 不在資料路徑上，它把 376-256-256-17 的策略堆疊寫成同一份層規格 — 雙擊可以在「模型架構」編輯器裡讀完整架構。優化器吃的是 `ppo-1.model`，也就是真正被訓練的那個 agent 網路。',
  },
  'Model_Architecture/ResNet-SkipConnection-CNN': {
    description:
      'mini-ResNet 的前向傳播：輸入是 CIFAR10 形狀的批次。每個層節點都作用在 TensorCreate 流出來的真實張量上，所以每一段的形狀都看得到。兩個 Add 節點（b1_add、b2_add）把殘差捷徑明白畫在畫布上。這不是訓練圖 — 要訓練請用 Usage_Example/ResNet-CIFAR10。',
  },
  'Model_Architecture/Seq2Seq-Attention-RNN': {
    description:
      '前向傳播示範：Transformer 出現之前，類神經機器翻譯所用的 Bahdanau 式 attention seq-to-seq。encoder LSTM 把來源序列 (batch=2, src_len=10, d=16) 壓成帶上下文的狀態，decoder LSTM 產生目標端狀態 (batch=2, tgt_len=8, d=16)，MultiHeadAttention 讓 decoder 的每一步都去查詢整段 encoder 狀態，得到的隱藏狀態再投影成詞彙 logits（示範用 vocab=64）。',
  },
  'Model_Architecture/SwinTransformer-Transformer': {
    description:
      '簡化版 Swin Transformer stage 的前向傳播。Swin 的特點是階層式 patch merging 加上 shifted-window attention，讓複雜度隨像素數線性成長。這裡保留了階層與 patch embedding（Conv2d stride=4，再一次 stride=2 下採樣），但每個 stage 用的是一般的 TransformerEncoder，而不是 shifted-window attention — 後者需要的 cycle-shift 運算目前還沒有對應的畫布節點。最後把 patch 平均池化，輸出 10 類 logits。',
  },
  'Model_Architecture/TimeSeries-LSTM-RNN': {
    description:
      '前向傳播示範：用過去 24 筆觀測值預測下一個值 — 天氣、金融、IoT 預測的經典「回看 → 預測下一步」模式。輸入是 (batch=2, seq=24, features=1) 的批次，例如每小時一筆的感測器讀數；進入 2 層、hidden size 32 的 LSTM，取完整輸出序列 (B, 24, 32)，Flatten 成 (B, 768)，再用 Linear 投影成每筆一個純量預測值。',
  },
  'Model_Architecture/UNet-Segmentation-CNN': {
    description:
      '兩層式的經典 U-Net，完整接成一張圖：每一個 encoder Conv、每一次 MaxPool、每一個 ConvTranspose、每一條 Concat skip 都看得到。一張 32×32 RGB 影像的路徑是 Lv1 Conv→ReLU 於 32×32（存下 skip）、MaxPool 到 16×16、Lv2 Conv→ReLU（存下 skip）、MaxPool 到 8×8、bottleneck Conv→ReLU，然後鏡像往上 — ConvTranspose 回 16×16、沿通道與 Lv2 的 skip 做 Concat、Conv→ReLU、ConvTranspose 回 32×32、與 Lv1 的 skip 做 Concat、Conv→ReLU，最後用 1×1 的 head Conv 投影成單一分割通道。encoder→decoder 的 skip-Concat 連線橫跨畫布，那個 U 形就是這個架構名字的由來。',
  },
  'Model_Architecture/ViT-ImageClassifier-Transformer': {
    description:
      'Vision Transformer 的前向傳播。(2,3,32,32) 的影像批次由 Conv2d stem（stride=8）切成互不重疊的 8x8 patch，得到 (2,48,4,4) 張量。Reshape 加 Permute 把空間格點攤平成 16 個 token 的序列，每個 token 48 維（seq=16, batch=2, d=48）。TransformerEncoder 對所有 patch 做全域 self-attention，最後把 patch token 平均池化，投影成 10 類 logits。',
  },

  // ── plugin: foundations -- Foundations teaching pack (C1, C2) ──
  'plugin:foundations/C1-3/Kernel-Effects': {
    description:
      '運算從頭到尾沒變，變的只有那九個權重 — §C1.3.4.1 的這句話在這裡用四個 Conv2dExplicit 檢查（圖 C1-3-e 的可執行版）。影像 (1, 1, 8, 8) 是手工排的：第 0–3 欄 100、第 4–7 欄 200（中間一條垂直邊），第 6 列第 1 欄再放一個 200 亮點（暗區裡的孤立點）。EdgeDetection 在亮點上剛好給 800（§C1.3.4.1 例 3-2 手算的數字），兩塊平坦區都是 0；Sharpen 讓平坦區維持原值（100 還是 100、200 還是 200），亮點升到 600；VerticalEdge 在兩條邊界欄給 300、亮點給 0，因為孤立點不是垂直邊；模糊是 Custom preset 九格全填 1/9，平坦區不動、亮點抹成 111.11、邊界軟化成 133 → 167 的斜坡。換 preset 或改 Custom 矩陣，只有輸出會變，圖不用動。',
  },
  'plugin:foundations/C2-1/Supervised-Learning-101': {
    description:
      '最小的「資料 → 模型 → 預測」迴圈。SyntheticDataset（kind=\'blobs\'、2 個中心）生出線性可分的點雲，TrainTestSplit 留 20% 當沒看過的考題，LogisticRegression 在訓練集上學 w·x+b，Accuracy 量它在保留集上還準不準。照原樣跑會看到 ≈95% 測試準確率 — 學的是規律，不是答案。換掉 SyntheticDataset 的 seed，換一批新資料也會得到同一類結果，這就是 §C2.1.4 說的訓練/測試紀律。',
  },
  'plugin:foundations/C2-2/Concentric-Circles-Failure': {
    description:
      'LogisticRegression 在這份同心圓資料上只拿到 ~50%，跟丟銅板一樣：內外兩圈，沒有任何一條直線切得開。資料由 SyntheticDataset(kind=\'circles\') 生成，Accuracy 負責量。把分類器換成 SVMClassifier、kernel=\'rbf\'，同樣的接線立刻跳到 ~95% — 這是 §C2.2.5 留下的伏筆，也是 C2-3（kernel trick）與 C2-4（神經網路）的動機。',
  },
  'plugin:foundations/C2-3/Decision-Tree-Iris': {
    description:
      '在 Iris（3 種 × 50 朵 × 4 個特徵）上 80/20 切分，訓練一棵深度 3 的決策樹 — 小到畫得上黑板，又夠深到拿下 ≈97% 測試準確率。每個內部節點只拿一個特徵跟一個門檻值比大小，由上往下讀就是 §C2.3.3 的二十問。把 max_depth 調到 8，訓練準確率會爬到 100%、測試準確率反而可能掉 — 最典型的過擬合症狀。',
  },
  'plugin:foundations/C2-3/SVM-RBF-Beats-Circles': {
    description:
      '同心圓換成 SVMClassifier、kernel=\'rbf\' 就從 50% 跳到 ≈95%。接線與 C2-2 Concentric-Circles-Failure 完全相同，只換這一個節點：RBF 把同心圓抬到三維、在那裡切一刀平面，投影回來就是一條閉合曲線。§C2.3.2 的 kernel trick 在這裡只是一個參數的差別。把 kernel 改成 \'linear\' 驗證：線性 SVM 一樣撞回 50% 那面牆。',
  },
  'plugin:foundations/C2-4/MLP-Solves-Circles': {
    description:
      '同一份同心圓資料，分類器換成兩層、帶 ReLU 的 MLP，準確率就跳到 ≈95% — 在 C2-2 打垮 LogisticRegression 的正是這份資料。§C2.4.4 的「多層 + 非線性」在這裡只是一張圖：輸入不變、換掉分類器節點，失敗就翻成成功。搭配下一張 C2-4 MLP-Without-Activation 一起看，就知道「深度」其實是激活函數給的。',
  },
  'plugin:foundations/C2-4/MLP-Without-Activation': {
    description:
      '架構與 MLP-Solves-Circles 相同（隱藏層 16, 16），只把 activation 設成 \'identity\'。層與層之間沒有非線性，兩個線性轉換就塌成一個：W₂·W₁·x + b 仍然只是一條直線。準確率掉回 50% 那面牆，跟 C2-2 的 LogisticRegression 一模一樣。兩張 C2-4 的圖並排跑，§C2.4.3 的「沒有激活，整個網路就是一層線性」就不再只是一句話。',
  },
  'plugin:foundations/C2-5/MLP-Inline-Demo': {
    description:
      'MLPClassifier 接一份合成的雙月資料，一秒內跑完，是 MLP-MNIST-Training 的替代版。網路仍是同一家族（多層 ReLU MLP，用 Adam 擬合），資料則是 200 個點而不是 60,000 張影像。適合快速確認引擎有沒有問題，也說明 MLP 跟 MNIST 沒有綁定 — 同一個模型，換任何表格資料都能跑。',
  },
  'plugin:foundations/C2-5/MLP-MNIST-Training': {
    description:
      '5 個 epoch 後驗證準確率 ≈98%、loss 從 ~2.3（隨機）掉到 ~0.05，對得上 §C2.5.4 的表。SequentialModel 組出 §C2.5.2 的架構（Flatten → Linear 784→128 → ReLU → Linear 128→64 → ReLU → Linear 64→10），Training Pipeline preset 接上標準的 Dataset(MNIST) → DataLoader → CrossEntropyLoss → Adam → TrainingLoop。把 preset 的 Epochs 調到 20 可以看完整的 §C2.5.4 訓練曲線；層設定裡的 activation 改成 identity，就會示範 §C2.5.6 的「沒有激活就只剩一層線性」。',
  },
  'plugin:foundations/Classical/Column-Stats-101': {
    description:
      '一張 6×3 的隨機表格進 EduColumnStats，檢視器的 Steps 分頁把每一欄的 sum → 除法 → 偏差平方 → 變異數 → 開根號逐步攤開 — 就是學生會寫在白板上的那條算式鏈。對應 C1-2（表格資料）。切換 unbiased 參數可以比較母體標準差與樣本標準差。',
  },
  'plugin:foundations/Classical/KNN-from-Scratch': {
    description:
      '只挑 petal length × petal width 兩個特徵（最能分開類別的一組），結果就畫得成節點內嵌的 2D 散佈圖。接續表格資料流程，80/20 分層切分之後，EduKNN 對 k=5 個最近的訓練點做多數決，距離用 euclidean。散佈圖把訓練點依類別上色、每個查詢點標上 `?`，可以直接用眼睛比對投票結果跟看得見的群聚形狀合不合。',
  },
  'plugin:foundations/Classical/Linear-Logistic-Compare': {
    description:
      '上面一條把隨機的 30×2 特徵與實數目標餵給 EduLinearRegression；下面一條讀 Iris 的兩個花瓣欄位，用 EduLogisticRegression 對物種標籤做分類。兩條鏈共用「特徵 → 線性組合」這副骨架，差別只在線性輸出之後那一步，這就是 C2-2 的重點。兩個 Print 節點各印一種模型的輸出，可以直接對照。',
  },

  // ── plugin: deep -- Deep Models teaching pack (C3, C4, C6) ──
  'plugin:deep/C3-1/Conv2D-Kernel-Effects': {
    description:
      '32 個 kernel 的 Conv2d 對 (1, 1, 8, 8) 合成影像做一次前向 — §C3.1.3.1「kernel 滑過去、輸出是逐元素相乘再相加」在張量形狀上的具體樣子。輸出 (1, 32, 8, 8) 是 32 個隨機初始化的 kernel 同時算出的 32 張特徵圖，也就是 §C3.1.4 的多 kernel 層。kernel 隨機是刻意的：要靠訓練才會學到邊緣／紋理／部件（§C3.1.4）。把 out_channels 調大，一層的參數量會照 `out_channels × in_channels × kernel² + out_channels`（最後一項是偏置）成長。',
  },
  'plugin:deep/C3-1/LeNet-MNIST-Training': {
    description:
      '§C3.1.5 的 LeNet 管線，跑成一次真的訓練。SequentialModel = Conv(32, 3×3) → ReLU → MaxPool → Conv(64, 3×3) → ReLU → MaxPool → Flatten → Linear(3136→128) → ReLU → Linear(128→10)；Training Pipeline preset 接上 Dataset(MNIST) → DataLoader → CrossEntropy → Adam → 5 個 epoch。驗證準確率約 99% — 學出來的 kernel（初始隨機 → 訓練後變成邊緣／紋理／部件，§C3.1.4）在 MNIST 上的標準示範。跟 C2-5 MLP-MNIST（沒有空間歸納偏置，約 98%）對照，就能感覺到 §C2.5.6 說的 MLP 參數爆炸，正是 CNN 解掉的問題。',
  },
  'plugin:deep/C3-2/UNet-Forward-Shapes': {
    description:
      '(1, 1, 16, 16) 合成影像走過一個最小的 U 形：兩組 Conv+Pool 把空間從 16→8→4 縮下去，ConvTranspose2d 再把 4→8 放回來，Concat 接上 §C3.2.2.3 對應編碼器層的 skip（8×8 + 8×8，skip 那一路 32 通道）。最後一個 ConvTranspose2d 回到 16×16。每一步印出來的形狀，就是課本 §C3.2.5「空間減半、通道加倍」換成真的張量數字 — 不訓練，只驗證幾何。要做真的分割訓練，把輸入換成 HuggingFaceDataset，再加一個 Training Pipeline preset。',
  },
  'plugin:deep/C3-3/Diffusion-Denoise-Loop': {
    description:
      '§C3.3.2 的反向去噪，整條接在一張圖上。GaussianNoise 抽出起點 x_T（1×3×16×16 標準常態），DiffusionUNet 組出預測噪聲的 U-Net（權重隨機 — 真正的 diffusion 模型要先在數百萬張影像上訓練過），DDPMSampler 用線性 beta 排程迭代 `num_steps=20` 步。輸出 x_0 形狀不變：模型訓練過的話這裡會是一張連貫的影像，權重隨機就只是一片被抹平的噪聲 — 但形狀的流動和 Stable Diffusion 推論時完全一樣（§C3.3.5）。把 GaussianNoise 的形狀改成 1,3,64,64，就能感覺到計算量怎麼隨解析度長大。',
  },
  'plugin:deep/C4-1/LSTM-Sequence-Forward': {
    description:
      'LSTM 在 (1, 10, 32) 張量上的前向：batch 1、序列長度 10、每個 token 32 維特徵。§C4.1.2「每一步兩次矩陣乘法加 tanh」在背後跑了 10 次；LSTM 另外還維持 §C4.1.5 的 cell state 高速公路，讓記憶撐過整條序列。輸出是逐步隱藏狀態 (1, 10, 64) 與最後的 h_T (1, 1, 64)。檢視器可以看 h_t 隨時間步變化；把最後的隱藏狀態印出來，就是 §C4.1.3.1「h_5 裝著整句話」的具體張量。',
  },
  'plugin:deep/C4-2/Co-Reference-Attention': {
    description:
      '§C4.2.4「她去了那家貓咖啡因為它很可愛」這個例句，接成一條跑得動的流程。TextInput 放句子，Tokenizer 切成 BPE 片段，EduTokenEmbedding 把每片抬成向量，PositionalEncoding 補上順序，EduSelfAttention 算出 (T, T) 注意力矩陣。檢視器會畫成熱圖：訓練夠久的 attention 會把代名詞接回它指的對象，「它」那一列的最大權重就落在「貓」上。這裡權重是隨機的，圖樣只是雜訊 — 要看的是 §C4.2.2 的三步演算法（內積 → softmax → 加權和），不是 attention「應該」說什麼。',
  },
  'plugin:deep/C4-3/Transformer-Block-Assembled': {
    description:
      '§C4.3.5 的 pre-norm Transformer block，用你已經學過的零件疊出來。(1, 5, 8) 張量依序走過 LayerNorm → EduMultiHeadAttention → Add（殘差接回原本的輸入）→ LayerNorm → EduFFN → Add。每個輸出埠都是 (1, 5, 8) — 一個 block 純粹是對 token 特徵做「原地增強」（§C4.3.6）。堆 N 層（GPT-2 small 是 12、Llama 70B 是 80）就長成實務上的架構；沒有新的運算子，就是同一個 block 重複。',
  },
  'plugin:deep/C4-4/LLM-Inference-Pipeline': {
    description:
      '§C4.4.2「訓練產生基礎模型、推論拿來用」的推論這一半。TextInput 放 prompt，Tokenizer 切成 BPE token（cl100k_base，和 GPT-4 同一套），EduTokenEmbedding 把每個 token id 換成向量，PositionalEncoding 補上位置，接著一條 Transformer block 形狀的鏈（LayerNorm → EduSelfAttention → EduFFN）扮演真實 LLM 80 幾層裡的其中一層。輸出形狀 (1, T, 8) — 真正的 LLM 會把它投影成 vocab logits 再 softmax，挑出下一個字。這裡權重隨機，出來的只是隨機隱藏狀態；§C4.4.3「用網路上的資料預訓練 → QA 對 SFT → 偏好 RLHF」那條路才是把知識填進這些權重的過程。',
  },
  'plugin:deep/C6-1/World-Model-Next-State': {
    description:
      '§C6.1「智能體想像接下來會發生什麼」縮到最小的樣子。把 4 維狀態和 2 維動作 Concat 成 6 維的 context，過 Linear → ReLU → Linear，出來就是預測的 4 維下一個狀態。真正的世界模型（Sora、DreamerV3、Genie）是拿 1080p 畫面做 30 個時間步，運算子完全一樣。把好幾個這種節點串起來就是 rollout — 模型不碰真實環境，自己「作夢」出一段多步的未來。',
  },
  'plugin:deep/C6-2/ViT-Full-Forward': {
    description:
      '把 Patchify-101 往下接完 §C6.2.4 的 ViT 流程。(1, 3, 16, 16) 影像由 EduPatchify 以 patch_size=4 切成 16 個 token；Squeeze 去掉 batch 維，讓 Edu 系列的 attention/FFN 節點（吃 [seq, D] 或 [seq, batch, D]）看到乾淨的 [16, 48] 序列。PositionalEncoding 加上位置向量，LayerNorm → EduMultiHeadAttention → EduFFN 就是一個 Transformer block（真的 ViT-Base 疊 12 個）。輸出 (16, 48) — 16 個 patch token、每個 48 維隱藏狀態 — 就是 ViT 餵給最後分類頭的東西。patch_size 調成 2 會變成 64 個 token，調成 8 則剩 4 個。',
  },
  'plugin:deep/C6-3/VLM-Cross-Modal-Attention': {
    description:
      '§C6.3「視覺 + 語言融合」的核心。9 個影像 patch token（12×12 影像經 EduPatchify，squeeze 成 [9, 48]）當 query，4 個文字 token（假造的 [4, 48] 張量）當 key/value。EduCrossAttention 的 Q 來自 query、K 與 V 來自 context — 每個影像 patch 去看哪個文字 token 跟自己最相關。輸出 (9, 48)：影像 token 數量不變，但每一個都帶上了文字的資訊。Flamingo、LLaVA、Claude Vision 底下都是這個運算子；把文字換成動作 token，就是 RT-2 這類 VLA 模型。',
  },
  'plugin:deep/C6-4/MoE-Routing': {
    description:
      '§C6.4.1「分組報告」那個比喻換成真的張量。輸入 (1, 8, 32)，也就是 8 個寬度 32 的 token。MoELayer 設 num_experts=4、top_k=2：每個 token 的 router 對 4 個專家打分，取前 2 名做 softmax，只合併這兩個專家的輸出。routing_weights (1, 8, 2) 和 expert_indices (1, 8, 2) 就是每個 token 挑了誰、各佔多少權重。這就是 §C6.4.2「總參數量很大、每個 token 的計算量很小」的取捨，DeepSeek-V3 671B（啟用 37B）和 Mixtral 都靠它。',
  },
  'plugin:deep/Diffusion/Cross-Attention-101': {
    description:
      'Stable Diffusion 的 U-Net 之所以能被文字 prompt 條件化，靠的就是這個機制。Self-attention 的 Q、K、V 全來自同一個張量；cross-attention 的 Q 來自「query」張量（影像 patch），K/V 來自「context」張量（文字 token 嵌入）。兩邊的序列長度可以不同 — 所以右邊那張熱圖是長方形的，列是影像位置、欄是文字位置。跑一次，看內嵌熱圖就知道每個「影像」位置去看了哪個「文字」位置。',
  },
  'plugin:deep/Diffusion/Mini-UNet-Expanded': {
    description:
      'diffusion U-Net 全部手工接線，每一個編碼器／解碼器 ResBlock、每一條 skip 都攤在畫布上。前向示範：一張帶噪的 16×16 RGB 影像經過 stem、兩段下行（各是 ResBlock + MaxPool，帶時間條件）、bottleneck，再兩段上行（各是 Upsample + Concat 接 skip + ResBlock），最後一個 1×1 卷積頭把通道收回 3。把 Concat 的 skip 邊（畫布上那些虛線）跟 ResNet preset 的 Add skip 對照 — 想法一樣，合併的運算不同。Compact preset 把同一套架構包進單一個 `DiffusionUNet` 節點。',
  },
  'plugin:deep/LLM/Multi-Head-Causal': {
    description:
      '兩個 attention head 加上 causal mask — GPT 用來預測下一個 token 的同一塊積木。每個 head 會學到自己的圖樣，熱圖一個 head 一張，長得都不一樣。causal mask 把上三角抹掉，所以位置 i 只能看到 ≤ i 的位置，偷看不到未來。attention 之後，EduFFN 把每個 token 的表示先放大再收回。拿 Self-Attention-101 對照：同一條鏈，這裡多了遮罩，而且複製成多個 head。',
  },
  'plugin:deep/LLM/Self-Attention-101': {
    description:
      '一個 attention head，整條攤開來看：文字 → token → 小維度 embedding → 位置編碼 → 手寫的 scaled dot-product attention。EduSelfAttention 上的熱圖就是 weights[i, j] — 位置 i 對位置 j 放了多少注意力。跑一次，再把 `causal` 改成 true 看上三角被擋掉，或調 `temperature` 看分布變尖或變平。',
  },
  'plugin:deep/Transformer/Patchify-101': {
    description:
      '一張合成的 1×3×16×16 影像，由 EduPatchify 切成 4×4 的 patch — C6-2（ViT）課程的前置。patch_size=4 時，16×16 影像變成 4×4 的 patch 網格 → 16 個長度 C·P·P = 48 的 token 向量。把 `flatten` 關掉可以保留每個 patch 的 [C,P,P] 結構，方便檢視器把每個 patch 畫成小圖。開啟 verbose 後，檢視器的「步驟」分頁會逐步走過 unfold → permute → flatten，看得到空間維度是怎麼塌進 token 維度的。',
  },

  // ── plugin: rl -- Reinforcement Learning teaching pack (C5) ──
  'plugin:rl/C5-1/RL-Trajectory-Mockup': {
    description:
      '這章是 L1，沒有演算法要跑，只把一條軌跡的五個要素（§C5.1.2）攤成張量。state (4, 6) 每列一個時間步、6 維觀測（像 CartPole）；action (4,) 每步一個整數（例如左、右、不動）；reward (4,) 每步一個浮點數。Mean 把「這回合整體好不好」收成一個數字，也就是 §C5.1.3 的回報。要換成真環境，把 state 的 TensorInput 換成 EnvWrapper、action 換成策略網路。',
  },
  'plugin:rl/C5-3/RLHF-Reward-Model': {
    description:
      'RewardModel 這個小 MLP 頭，把 128 維隱藏狀態壓成一個純量獎勵。兩個 TensorInput 代表 chosen 與 rejected 兩份 LLM 輸出的隱藏狀態，也就是 §C5.3.3 的「兩塔比較」。用人類偏好資料訓練過之後 chosen 應該比 rejected 高；這裡是隨機初始化，分數就只是亂數。真正把「chosen − rejected」推成正值的，是 §C5.3.4 的 Bradley-Terry log-sigmoid 損失。',
  },
  'plugin:rl/C5-4/GRPO-Group-Advantage': {
    description:
      'DeepSeek-R1 §C5.4 的核心技巧：同一個 prompt 生成 K=8 個候選回答（這裡直接拿 8 個獎勵分數代替），組內平均就是 §C5.4.2 的基準，reward − 組平均就是每個回答的優勢。PPO 的基準要另外訓練一個 critic，GRPO 只要一個算術平均 — R1 訓練成本之所以低，靠的就是這一步簡化。優勢為負（比組平均差）的回答會被策略梯度壓下去，為正的則被強化。',
  },
  'plugin:rl/RL/Policy-Gradient-101': {
    description:
      '三個輸入張量代替一條軌跡：logits（策略對 3 個動作、4 個時間步的偏好）、actions（實際抽到哪個動作）、rewards（後續拿到的回報）。EduPolicyGradient 依序做 softmax、取出 π(a|s)、減掉批次平均基準得到優勢，最後組出純量損失 — 也就是 C5-2（PPO）拿掉 clip 的那一半。跑完後把 baseline 改成 none，看看拿掉這個降低變異數的技巧後損失會怎麼變。',
  },

  // ── plugin: stats -- Stats teaching pack ──
  'plugin:stats/Stats/Confusion-Matrix-Heatmap': {
    description:
      '用 iris 的 70% 訓練一棵淺決策樹，對保留的 30% 做預測，再把預測與真實標籤送進 Stats-ConfusionMatrix。矩陣會以標好類別名稱的熱圖出現在結果面板與節點詳情裡，versicolor 和 virginica 混淆時就是非對角線上亮起來的那一格，不必自己在數字裡找。把 normalize 改成 true，對角線上讀到的就是每一類的召回率。',
  },
  'plugin:stats/Stats/Iris-Describe-Table': {
    description:
      '讀進隨附的 iris.csv 交給 Stats-Describe，輸出和 pandas 的 describe() 一致。每個數值欄位給出 count、mean、std、min、25/50/75%、max，Stats-TableView 再把結果印在結果面板，左側帶著統計量名稱。從一個 CSV 到「知道裡面有什麼」，這是最短的一條路。',
  },
  'plugin:stats/Stats/Iris-GroupBy-Chart': {
    description:
      'CSV → GroupBy → Chart。species 欄位當分組鍵（CSVReader 的 labels 輸出正好就是 Stats-GroupByAggregate 要的逐列標籤），每個數值欄位依品種取平均，Stats-ChartView 再把平均花瓣長度畫成一個品種一根長條。同一張分組表格會在旁邊以文字印出來，圖和數字擺在一起對得上。',
  },
};

export default zhTW;
