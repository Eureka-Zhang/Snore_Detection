# predict_snore.py
import torch
import torch.nn as nn
import numpy as np
import sys
import os
from preprocess import preprocess_audio
from train_cnn_lstm import CNN_LSTM

# =====================
# 预测函数
# =====================
def predict_snore(model_path, audio_path):
    # 1. 加载模型
    model = CNN_LSTM()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # 2. 预处理音频
    cqt_feat = preprocess_audio(audio_path)
    tensor = torch.tensor(cqt_feat, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,F,T]

    # 3. 推理
    with torch.no_grad():
        output = model(tensor)
        prob = output[0].numpy()
        pred_class = np.argmax(prob)
        label = "Snore 💤" if pred_class == 1 else "Non-snore 😴"
        confidence = prob[pred_class] * 100

    # 4. 打印结果
    print(f"\n🎧 文件: {os.path.basename(audio_path)}")
    print(f"🔍 预测结果: {label}")
    print(f"📊 置信度: {confidence:.2f}%\n")

    return label, confidence


# =====================
# 命令行入口
# =====================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法示例: python predict_snore.py data/snore_test.wav")
        sys.exit(0)

    audio_file = sys.argv[1]
    model_path = "snore_cnn_lstm.pth"

    if not os.path.exists(model_path):
        print("❌ 找不到模型文件 snore_cnn_lstm.pth，请先运行 train_cnn_lstm.py")
        sys.exit(0)

    predict_snore(model_path, audio_file)