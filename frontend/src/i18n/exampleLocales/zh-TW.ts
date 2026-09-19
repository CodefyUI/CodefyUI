import type { ExampleTranslations } from './types';

/**
 * Traditional Chinese for the example gallery, keyed by the `path`
 * `/api/examples/list` returns.
 *
 * Names are absent on purpose: an example's English name is its handle in the
 * docs, in `examples/` on disk and in a `run_graph.py` argument, so it reads
 * the same in both locales and only the description changes.
 *
 * A description is one line of at most 56 columns -- what a card, a sidebar
 * row and the detail pane show without cutting -- saying what the graph shows
 * plus any requirement that stops a run: a GPU, a download, a pack that has to
 * be installed first. Everything longer belongs in a note on the canvas,
 * beside the nodes it is about. `exampleLocales/zh-TW.test.ts` pins the rule.
 */
const zhTW: ExampleTranslations = {
  // ── Usage examples -- the beginner-facing starters ──
  'Usage_Example/Api-Function': {
    description: '把訊息原樣回傳：在畫布上執行，或用 HTTP POST 呼叫',
  },
  'Usage_Example/CNN-MNIST/InferenceCNN-MNIST': {
    description: '辨識一個數字；需先跑過 Train CNN on MNIST',
  },
  'Usage_Example/CNN-MNIST/TrainCNN-MNIST': {
    description: '在 MNIST 上訓練 CNN 再測試；需要下載',
  },
  'Usage_Example/GPT-Mini/TrainGPT-Mini': {
    description: '每張影像 16 個 token，再看測試準確率；需要下載',
  },
  'Usage_Example/HuggingFace-Dataset/TrainCNN-Beans': {
    description: '三類豆葉影像的測試準確率；需下載',
  },
  'Usage_Example/ResNet-CIFAR10/TrainResNet-CIFAR10': {
    description: '訓練迷你 ResNet 並評分；需下載 170 MB',
  },
  'Usage_Example/ResNet18-CIFAR10-Baseline': {
    description: '測試準確率約 95%；需 GPU 與下載',
  },

  // ── Classical ML ──
  'Classical/Iris-Sklearn-KNN': {
    description: '用 sklearn KNN 分類 Iris，印出測試準確率',
  },
  'Classical/Tabular-Iris-Pipeline': {
    description: '每個特徵欄位的平均值都落在 0 附近',
  },

  // ── LLM -- embeddings, retrieval, causal-LM training ──
  'LLM/RAG-LLMChat-API': {
    description: '需 sentence-embeddings 套件包與一個跑著的 Ollama',
  },
  'LLM/RAG-Local-Offline': {
    description: '離線作答；需 rag 與 sentence-embeddings 套件包',
  },
  'LLM/Sentence-Similarity-zhTW': {
    description: '配對八個句子；需 sentence-embeddings 套件包',
  },
  'LLM/TrainCausalLM-TinyStories': {
    description: '困惑度與一段它寫的故事；需 16 GB GPU 與下載',
  },
  'LLM/Word-Embedding-Analogy': {
    description: '在 59 字的玩具詞向量表上做加減，離線可跑',
  },

  // ── Diffusion ──
  'Diffusion/Forward-Process': {
    description: '把高斯噪聲混進一張真實的手寫數字圖，再看結果',
  },
  'Diffusion/Mini-UNet-Compact': {
    description: '跑一步去噪；輸出形狀與輸入相同',
  },
  'Diffusion/Toy-Sampling': {
    description: '跑 20 個反向步驟；模型沒訓練過，輸出是噪聲',
  },

  // ── Transformer ──
  'Transformer/MoE-TopK-Routing': {
    description: '4 位專家中挑 2 位的路由，逐 token 印出',
  },

  // ── RNN ──
  'RNN/RNN-OneStep': {
    description: '印出整段序列讀完之後的 hidden state',
  },

  // ── Reinforcement learning ──
  'RL/RLHF-Reward-and-KL': {
    description: '每條序列一個獎勵分數，再加上對參考模型的 KL',
  },

  // ── Vision-language-action ──
  'VLA/TrainVLA-PushWorld': {
    description: '閉環成功率 0.97；需 GPU、約一小時',
  },

  // ── Model architectures -- illustrative forward passes ──
  'Model_Architecture/BERT-Encoder-Transformer': {
    description: '編碼器前向傳播：每個 token 都能注意到全部',
  },
  'Model_Architecture/BiGRU-SpeechRecognition-RNN': {
    description: '雙向 GRU 前向傳播，輸入是 mel 頻譜幀',
  },
  'Model_Architecture/ConvNeXt-CNN': {
    description: 'ConvNeXt 區塊前向傳播：patchify stem 與 1x1 卷積',
  },
  'Model_Architecture/DQN-Atari-RL': {
    description: '前向傳播：四張堆疊畫面換四個動作的 Q 值',
  },
  'Model_Architecture/DiT-Diffusion-Transformer': {
    description: '前向傳播：帶噪 patch 進去，預測的噪聲出來',
  },
  'Model_Architecture/EfficientNet-CNN': {
    description: 'MBConv 前向傳播：擴張、squeeze-excite、投影',
  },
  'Model_Architecture/GPT-DecoderOnly-Transformer': {
    description: '只有 decoder 的 Transformer 前向傳播到 logits',
  },
  'Model_Architecture/LLaMA-Decoder-Transformer': {
    description: 'decoder 前向傳播，attention 之前先做 LayerNorm',
  },
  'Model_Architecture/PPO-Robotics-RL': {
    description: 'actor-critic 前向傳播：376 個輸入換 17 個動作',
  },
  'Model_Architecture/ResNet-SkipConnection-CNN': {
    description: 'mini-ResNet 前向傳播，捷徑就是兩個 Add 節點',
  },
  'Model_Architecture/Seq2Seq-Attention-RNN': {
    description: '編碼器-解碼器 LSTM 前向傳播，中間接 attention',
  },
  'Model_Architecture/SwinTransformer-Transformer': {
    description: '兩層階層式前向傳播，中間做一次 patch merging',
  },
  'Model_Architecture/TimeSeries-LSTM-RNN': {
    description: 'LSTM 前向傳播：24 步歷史換一個預測值',
  },
  'Model_Architecture/UNet-Segmentation-CNN': {
    description: 'U-Net 前向傳播，skip 連線就是 Concat 節點',
  },
  'Model_Architecture/ViT-ImageClassifier-Transformer': {
    description: '前向傳播：影像切成 16 個 patch token 再分類',
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
