import type { ExampleTranslations } from './types';

/**
 * Traditional Chinese for the example gallery, keyed by the `path`
 * `/api/examples/list` returns.
 *
 * Names are absent on purpose: an example's English name is its handle in the
 * docs, in `examples/` on disk and in a `run_graph.py` argument, so it reads
 * the same in both locales and only the description changes.
 *
 * A description is one line of at most 40 columns -- what a card, a sidebar
 * row and the detail pane show without cutting -- saying what the graph does
 * and what it is explaining, never what a run needs. A download, a GPU, a
 * pack, a key, a sibling example to run first, and every longer explanation
 * live in a canvas note. `exampleLocales/zh-TW.test.ts` pins the rule.
 */
const zhTW: ExampleTranslations = {
  // ── Usage examples -- the beginner-facing starters ──
  'Usage_Example/Api-Function': {
    description: '把 POST body 裡的訊息原樣回傳',
  },
  'Usage_Example/CNN-MNIST/InferenceCNN-MNIST': {
    description: '載入存好的權重，認出一個數字',
  },
  'Usage_Example/CNN-MNIST/TrainCNN-MNIST': {
    description: '60,000 張訓練，10,000 張測試',
  },
  'Usage_Example/GPT-Mini/TrainGPT-Mini': {
    description: '注意力用在影像，不只在文字',
  },
  'Usage_Example/HuggingFace-Dataset/TrainCNN-Beans': {
    description: '資料集的每一段都是一顆節點',
  },
  'Usage_Example/ResNet-CIFAR10/TrainResNet-CIFAR10': {
    description: '每個區塊只學一個修正量',
  },
  'Usage_Example/ResNet18-CIFAR10-Baseline': {
    description: '每一層都在編輯器裡，不寫 Python',
  },
  'Usage_Example/Segmentation-Synthetic-Shapes': {
    description: '替每個像素標上圓形、方形或背景',
  },

  // ── Classical ML ──
  'Classical/Iris-Sklearn-KNN': {
    description: '兩個花瓣欄位，五個鄰居投票',
  },
  'Classical/Tabular-Iris-Pipeline': {
    description: '每個特徵的平均值最後都接近 0',
  },
  'Classical/Classifier-Showdown-Moons': {
    description: '哪一種邊界彎得夠',
  },

  // ── LLM -- embeddings, retrieval, causal-LM training ──
  'LLM/RAG-LLMChat-API': {
    description: '檢索在本機，生成不在本機',
  },
  'LLM/RAG-Local-Offline': {
    description: '0.5B 模型靠五篇筆記作答',
  },
  'LLM/Sentence-Similarity-zhTW': {
    description: '四組句子意思相同，用字不同',
  },
  'LLM/TrainCausalLM-TinyStories': {
    description: '從隨機權重到一則短故事',
  },
  'LLM/Word-Embedding-Analogy': {
    description: 'King - man + woman 得到 queen',
  },

  // ── Diffusion ──
  'Diffusion/Forward-Process': {
    description: '加一步噪聲；那個 7 還看得出來',
  },
  'Diffusion/Mini-UNet-Compact': {
    description: '跑一步去噪，輸出形狀與輸入相同',
  },
  'Diffusion/Toy-Sampling': {
    description: '20 步；沒訓練過，輸出仍是噪聲',
  },
  'Diffusion/Train-DDPM-Blobs': {
    description: '先學會去噪，再取樣出新影像',
  },

  // ── Transformer ──
  'Transformer/MoE-TopK-Routing': {
    description: '每個 token 只跑 4 位專家中的 2 位',
  },
  'Transformer/Causal-Attention-From-Scratch': {
    description: 'Q K^T、縮放、遮罩、softmax、乘上 V',
  },

  // ── RNN ──
  'RNN/RNN-OneStep': {
    description: '同一組權重，每一步都重複使用',
  },
  'RNN/LSTM-Recall-First-Token': {
    description: '把一個 token 帶過 9 步雜訊',
  },
  'Papers/Gradient-Clipping-RNN': {
    description: '裁剪梯度後 RNN 就學得會',
  },
  'Papers/Autoencoder-MNIST-2D': {
    description: '每個數字壓成 2 個數，和 PCA 對照',
  },
  'Papers/LSTM-Forecast-Sine': {
    description: '預測波形的下一個值',
  },
  'Papers/Transformer-Sort': {
    description: '用自注意力排序 10 個數字',
  },
  'Papers/Weight-Decay-Regression': {
    description: '同一個網路，有無權重衰減',
  },
  'Papers/Early-Stopping-Regression': {
    description: '驗證損失回升時就停止訓練',
  },

  // ── Paper reproductions ──
  'Papers/Dropout-Srivastava-2014': {
    description: '同一個 MLP 加與不加 dropout 0.5',
  },
  'Papers/BatchNorm-Ioffe-2015': {
    description: 'SGD lr 0.5 下的 MLP，有無 BN',
  },
  'Papers/ResNet-Degradation-He-2016': {
    description: '20 層 MLP：一般與加捷徑',
  },
  'Papers/Adam-Kingma-2015': {
    description: '同一個 MLP 用三種最佳化器訓練',
  },
  'Papers/Label-Smoothing-Szegedy-2016': {
    description: '硬標籤與 label smoothing 0.1',
  },
  'Papers/SGDR-Loshchilov-2017': {
    description: '餘弦重啟與固定學習率',
  },
  'Papers/Vanishing-Gradients-Glorot-2010': {
    description: '深層 sigmoid MLP 與 ReLU 版',
  },

  // ── Reinforcement learning ──
  'RL/RLHF-Reward-and-KL': {
    description: '一個給答案打分，一個不讓策略跑遠',
  },
  'RL/Reward-Hacking-Shortcut': {
    description: '訓練集滿分，保留集卻被騙',
  },
  'RL/GRPO-Step-GridWorld': {
    description: '每一回合都和同組的回合比較',
  },

  // ── Paper reproductions -- reinforcement learning ──
  'Papers/REINFORCE-CartPole': {
    description: '直接用回報做策略梯度',
  },
  'Papers/DQN-CartPole': {
    description: '從經驗回放學出 Q 值',
  },
  'Papers/PPO-CartPole': {
    description: '裁切比值的策略更新加上 GAE',
  },
  'Papers/QLearning-GridWorld': {
    description: '用 Q 值繞過兩個陷阱走到終點',
  },

  // ── Vision-language-action ──
  'VLA/TrainVLA-PushWorld': {
    description: '該推哪一顆冰球，由指令決定',
  },

  // ── Model architectures -- illustrative forward passes ──
  'Model_Architecture/BERT-Encoder-Transformer': {
    description: '每個 token 都注意到其他每一個',
  },
  'Model_Architecture/BiGRU-SpeechRecognition-RNN': {
    description: '32 幀，每幀 50 個音素分數',
  },
  'Model_Architecture/ConvNeXt-CNN': {
    description: 'patchify stem、一個 7x7、兩個 1x1 卷積',
  },
  'Model_Architecture/DQN-Atari-RL': {
    description: '四張堆疊畫面換四個 Q 值',
  },
  'Model_Architecture/DiT-Diffusion-Transformer': {
    description: '帶噪 patch 進去，預測的噪聲出來',
  },
  'Model_Architecture/EfficientNet-CNN': {
    description: '先擴張，再 squeeze-excite，最後投影',
  },
  'Model_Architecture/GPT-DecoderOnly-Transformer': {
    description: '同一個張量同時當輸入與 memory',
  },
  'Model_Architecture/LLaMA-Decoder-Transformer': {
    description: '整個堆疊前先做一次 LayerNorm',
  },
  'Model_Architecture/PPO-Robotics-RL': {
    description: '376 個輸入換 17 個動作機率',
  },
  'Model_Architecture/ResNet-SkipConnection-CNN': {
    description: '捷徑就畫成 Add 節點',
  },
  'Model_Architecture/Seq2Seq-Attention-RNN': {
    description: '解碼器每一步都查詢 10 個狀態',
  },
  'Model_Architecture/SwinTransformer-Transformer': {
    description: '64 個 token 併成 16 個更寬的 token',
  },
  'Model_Architecture/TimeSeries-LSTM-RNN': {
    description: '24 筆歷史讀數，輸出一個數字',
  },
  'Model_Architecture/UNet-Segmentation-CNN': {
    description: '每個像素一個 logit，輸出 32x32',
  },
  'Model_Architecture/ViT-ImageClassifier-Transformer': {
    description: '一張影像變成 16 個 patch token',
  },

  // ── plugin: foundations -- Foundations teaching pack (C1, C2) ──
  'plugin:foundations/C1-3/Kernel-Effects': {
    description: '邊緣、銳化、垂直邊緣與模糊',
  },
  'plugin:foundations/C2-1/Supervised-Learning-101': {
    description: '用 160 個點訓練，保留的 40 個點評分',
  },
  'plugin:foundations/C2-2/Concentric-Circles-Failure': {
    description: '兩圈點，沒有直線切得開',
  },
  'plugin:foundations/C2-3/Decision-Tree-Iris': {
    description: '深度 3 的樹，印成一條條規則',
  },
  'plugin:foundations/C2-3/SVM-RBF-Beats-Circles': {
    description: '畫出來的邊界收成一個圈',
  },
  'plugin:foundations/C2-4/MLP-Solves-Circles': {
    description: '兩道 ReLU 轉折，把圈變成區域',
  },
  'plugin:foundations/C2-4/MLP-Without-Activation': {
    description: '兩層線性疊起來還是一層',
  },
  'plugin:foundations/C2-5/MLP-Inline-Demo': {
    description: '兩個月牙形；一顆節點擬合又分類',
  },
  'plugin:foundations/C2-5/MLP-MNIST-Training': {
    description: '影像攤平，再走 784-128-64-10',
  },
  'plugin:foundations/Classical/Column-Stats-101': {
    description: '逐欄印出平均、標準差與最小值',
  },
  'plugin:foundations/Classical/KNN-from-Scratch': {
    description: 'k=5 的多數決，畫成散佈圖',
  },
  'plugin:foundations/Classical/Linear-Logistic-Compare': {
    description: '同一套骨架，輸出數值或品種',
  },

  // ── plugin: deep -- Deep Models teaching pack (C3, C4, C6) ──
  'plugin:deep/C3-1/Conv2D-Kernel-Effects': {
    description: '8x8 影像變 32 張特徵圖，池化成 4x4',
  },
  'plugin:deep/C3-1/LeNet-MNIST-Training': {
    description: '5 個 epoch，測 10,000 個沒看過的數字',
  },
  'plugin:deep/C3-2/UNet-Forward-Shapes': {
    description: '16 縮到 4 再放大回來，跳接 48 通道',
  },
  'plugin:deep/C3-3/Diffusion-Denoise-Loop': {
    description: '20 步；U-Net 沒訓練過，輸出仍是噪聲',
  },
  'plugin:deep/C4-1/LSTM-Sequence-Forward': {
    description: '10 個 hidden state，最後一個另外印出',
  },
  'plugin:deep/C4-2/Co-Reference-Attention': {
    description: '14 個字、25 個 token、25x25 的圖',
  },
  'plugin:deep/C4-3/Transformer-Block-Assembled': {
    description: '六顆節點，(1, 5, 8) 進、(1, 5, 8) 出',
  },
  'plugin:deep/C4-4/LLM-Inference-Pipeline': {
    description: '一層：提示詞進，hidden state 出',
  },
  'plugin:deep/C6-1/World-Model-Next-State': {
    description: '輸入狀態與動作，輸出預測',
  },
  'plugin:deep/C6-2/ViT-Full-Forward': {
    description: '每塊 4x4 patch 變成 48 維 token',
  },
  'plugin:deep/C6-3/VLM-Cross-Modal-Attention': {
    description: '9 塊 patch 讀 4 個文字 token，4 個 head',
  },
  'plugin:deep/C6-4/MoE-Routing': {
    description: '逐個 token 印出專家編號與權重',
  },
  'plugin:deep/Diffusion/Cross-Attention-101': {
    description: '兩個 head，6 個 query 對上 4 個 key',
  },
  'plugin:deep/Diffusion/Mini-UNet-Expanded': {
    description: '每個區塊都看得到；skip 用 Concat 合併',
  },
  'plugin:deep/LLM/Multi-Head-Causal': {
    description: '兩張 5x5 的圖，遮罩處畫上斜線',
  },
  'plugin:deep/LLM/Self-Attention-101': {
    description: '6 個 token，一張誰讀誰的 6x6 圖',
  },
  'plugin:deep/Transformer/Patchify-101': {
    description: '影像變成 16 個 token，每個 48 個數字',
  },

  // ── plugin: rl -- Reinforcement Learning teaching pack (C5) ──
  'plugin:rl/C5-1/RL-Trajectory-Mockup': {
    description: '狀態、動作、獎勵與回報',
  },
  'plugin:rl/C5-3/RLHF-Reward-Model': {
    description: '未訓練的獎勵頭給兩份回答打分',
  },
  'plugin:rl/C5-4/GRPO-Group-Advantage': {
    description: '8 個獎勵與組內平均，不用 critic',
  },
  'plugin:rl/RL/Policy-Gradient-101': {
    description: '一步 REINFORCE：損失與優勢',
  },

  // ── plugin: eval -- Eval pack ──
  'plugin:eval/Eval/AB-Text-Classification': {
    description: '兩個情感分類模型逐筆比較',
  },
  'plugin:eval/Eval/VQA-Shapes': {
    description: '影像加問題，答案逐筆評分',
  },

  // ── plugin: stats -- Stats teaching pack ──
  'plugin:stats/Stats/Confusion-Matrix-Heatmap': {
    description: '真實對上預測，45 列 iris 資料',
  },
  'plugin:stats/Stats/Iris-Describe-Table': {
    description: '四個欄位，各八項統計量',
  },
  'plugin:stats/Stats/Iris-GroupBy-Chart': {
    description: '三根長條：平均花瓣長度',
  },

  // ── Data and IO examples (feat/examples-data-io) ──
  'Usage_Example/Augment-Then-Train': {
    description: '訓練時旋轉數字並調整明暗',
  },
  'Usage_Example/Resume-From-Checkpoint': {
    description: '10 個 epoch 後存檔，再接著訓練 10 個',
  },
  'Classical/KFold-Cross-Validation': {
    description: '三份資料、三個誤差、一個平均',
  },
  'Others/Reduce-And-Switch': {
    description: '選一個欄位，摺疊成一個數字',
  },
  'Others/Tensor-Ops-Tour': {
    description: '對同一個網格切分、堆疊、選列、池化',
  },

  // ── Built-ins added after the plugin blocks (papers, concepts, llm) ──
  'Papers/GroupNorm-Wu-2018': {
    description: '一次 2 張：BatchNorm 對上 GroupNorm',
  },
  'RNN/Conv1d-Sequence-Classifier': {
    description: '時間卷積用滑動視窗讀 token',
  },
  'RL/Reward-Model-Bradley-Terry': {
    description: '訓練前後的 Bradley-Terry 損失',
  },
  'RL/Discounted-Returns': {
    description: '一個獎勵、三個 gamma、三種回報',
  },
  'LLM/Data-Mix-Ratios': {
    description: '兩個語料：照原樣混合，再 1:1 混合',
  },
  'Others/One-Backward-Pass': {
    description: '梯度只會到達被查過的那幾列',
  },
  'Others/Image-Batch-IO': {
    description: '寫入 8 張影像，再讀回整個資料夾',
  },
};

export default zhTW;
