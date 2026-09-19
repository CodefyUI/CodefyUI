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
    description: '由 Ollama 作答；需 sentence-embeddings 套件包',
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
    description: 'decoder 前向傳播，整個堆疊前先做一次 LayerNorm',
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
    description: '兩階段階層式前向傳播，中間做一次 patch merging',
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
    description: '邊緣、銳化、垂直邊緣、模糊，四張輸出直接印出',
  },
  'plugin:foundations/C2-1/Supervised-Learning-101': {
    description: '用 160 個點訓練，再用保留的 40 個點評分',
  },
  'plugin:foundations/C2-2/Concentric-Circles-Failure': {
    description: '一條直線切兩圈點：60 個測試點只對 27 個',
  },
  'plugin:foundations/C2-3/Decision-Tree-Iris': {
    description: '深度 3 的樹分類 Iris，並印出學到的規則',
  },
  'plugin:foundations/C2-3/SVM-RBF-Beats-Circles': {
    description: '改用 RBF 核之後，60 個測試點全部答對',
  },
  'plugin:foundations/C2-4/MLP-Solves-Circles': {
    description: '兩層 ReLU，60 個測試點全部答對',
  },
  'plugin:foundations/C2-4/MLP-Without-Activation': {
    description: '拿掉 ReLU，同一個網路就掉回亂猜的水準',
  },
  'plugin:foundations/C2-5/MLP-Inline-Demo': {
    description: '兩個月牙形，用一個 MLP 節點大約一秒就分開',
  },
  'plugin:foundations/C2-5/MLP-MNIST-Training': {
    description: '在 MNIST 上訓練這個 MLP，再測準確率；需要下載',
  },
  'plugin:foundations/Classical/Column-Stats-101': {
    description: '逐欄印出平均、標準差與最小值',
  },
  'plugin:foundations/Classical/KNN-from-Scratch': {
    description: 'k=5 的多數決，直接畫在節點上的散佈圖',
  },
  'plugin:foundations/Classical/Linear-Logistic-Compare': {
    description: '上面那條預測數值，下面那條預測鳶尾花品種',
  },

  // ── plugin: deep -- Deep Models teaching pack (C3, C4, C6) ──
  'plugin:deep/C3-1/Conv2D-Kernel-Effects': {
    description: '一張 8x8 影像變成 32 張特徵圖，池化後剩 4x4',
  },
  'plugin:deep/C3-1/LeNet-MNIST-Training': {
    description: '在 MNIST 上訓練 LeNet CNN，再測準確率；需下載',
  },
  'plugin:deep/C3-2/UNet-Forward-Shapes': {
    description: '16 縮到 4 再放大回來，中間接一條 48 通道跳接',
  },
  'plugin:deep/C3-3/Diffusion-Denoise-Loop': {
    description: '未訓練的 U-Net 跑 20 個反向步驟；輸出是噪聲',
  },
  'plugin:deep/C4-1/LSTM-Sequence-Forward': {
    description: '十步的 hidden state，最後一個另外印出來',
  },
  'plugin:deep/C4-2/Co-Reference-Attention': {
    description: '14 個字切成 25 個 token，畫成 25x25 的圖；需下載',
  },
  'plugin:deep/C4-3/Transformer-Block-Assembled': {
    description: '六顆節點、一種形狀：(1, 5, 8) 進、(1, 5, 8) 出',
  },
  'plugin:deep/C4-4/LLM-Inference-Pipeline': {
    description: '五個 token 各自成為 8 維 hidden；需下載',
  },
  'plugin:deep/C6-1/World-Model-Next-State': {
    description: '輸入狀態與動作，輸出預測的下一個狀態',
  },
  'plugin:deep/C6-2/ViT-Full-Forward': {
    description: '每塊 4x4 的 patch 變成 16 個 48 維 token 之一',
  },
  'plugin:deep/C6-3/VLM-Cross-Modal-Attention': {
    description: '九個影像 patch 讀四個文字 token，共四個 head',
  },
  'plugin:deep/C6-4/MoE-Routing': {
    description: '逐個 token 印出權重和專家編號',
  },
  'plugin:deep/Diffusion/Cross-Attention-101': {
    description: '長方形的注意力圖：6 個 query 對上 4 個 key',
  },
  'plugin:deep/Diffusion/Mini-UNet-Expanded': {
    description: 'skip 用 Concat 合併，通道數在接點相加',
  },
  'plugin:deep/LLM/Multi-Head-Causal': {
    description: '兩張圖，對角線以上畫斜線；需下載',
  },
  'plugin:deep/LLM/Self-Attention-101': {
    description: '6 個 token 變成 6x6 的注意力圖；需下載',
  },
  'plugin:deep/Transformer/Patchify-101': {
    description: '一張影像變成 16 個 token，每個 48 個數字',
  },

  // ── plugin: rl -- Reinforcement Learning teaching pack (C5) ──
  'plugin:rl/C5-1/RL-Trajectory-Mockup': {
    description: '印出一個回合：狀態、動作、獎勵與回報',
  },
  'plugin:rl/C5-3/RLHF-Reward-Model': {
    description: '未訓練的獎勵頭給兩份回答打分數',
  },
  'plugin:rl/C5-4/GRPO-Group-Advantage': {
    description: '印出八個候選回答的獎勵，以及組內平均',
  },
  'plugin:rl/RL/Policy-Gradient-101': {
    description: '印出損失、優勢，以及每一步的 log pi(a|s)',
  },

  // ── plugin: stats -- Stats teaching pack ──
  'plugin:stats/Stats/Confusion-Matrix-Heatmap': {
    description: '45 列保留的 iris 資料，真實對上預測',
  },
  'plugin:stats/Stats/Iris-Describe-Table': {
    description: '四個 iris 欄位，各給八列摘要統計',
  },
  'plugin:stats/Stats/Iris-GroupBy-Chart': {
    description: '三根長條：每個 iris 品種的平均花瓣長度',
  },
};

export default zhTW;
