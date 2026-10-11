/**
 * Traditional Chinese for the presets the backend ships (#684), keyed by
 * `preset_name`.
 *
 * Only the description is translated. The name is the preset's handle -- the
 * palette, a saved graph and the Presets tab all key on it -- so it reads the
 * same in both locales, as node names do. A preset with no entry here, one a
 * user saved or a plugin ships, shows its own description.
 */
const zhTW: Record<string, { description?: string }> = {
  'LSTM Sequence': {
    description: 'LSTM 層後接一個激活函數。適合文字分類、時間序列等序列建模任務。',
  },
  'Simple CNN Classifier': {
    description:
      '基本的 CNN 分類器：Conv2d -> ReLU -> MaxPool2d -> Conv2d -> ReLU -> MaxPool2d。適合 MNIST 這類影像分類任務。',
  },
  'Training Pipeline': {
    description:
      '完整的訓練流程：Dataset -> DataLoader -> Loss -> Optimizer -> TrainingLoop。接上模型即可開始訓練。',
  },
};

export default zhTW;
