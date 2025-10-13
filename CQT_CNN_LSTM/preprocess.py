# preprocess.py
import os
import numpy as np
import librosa
import librosa.display

def preprocess_audio(audio_path, sr=48000, duration=10, hop_length=512, bins_per_octave=12, n_bins=84):
    #注意采样率问题
    """
    对音频进行预处理，输出CQT谱图
    """
    # 1. 读取音频
    y, sr = librosa.load(audio_path, sr=sr)
    
    # 2. 截取或补零到3.5秒
    target_len = int(sr * duration)
    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')

    # 3. 去除直流与低频噪声（50Hz高通滤波）
    y = librosa.effects.preemphasis(y)

    # 4. 计算CQT谱图
    cqt = librosa.cqt(
        y=y, sr=sr, hop_length=hop_length, bins_per_octave=bins_per_octave, n_bins=n_bins, window='hann'
    )
    cqt_abs = np.abs(cqt)
    
    # 5. 转换为对数幅度谱（dB）
    cqt_db = librosa.amplitude_to_db(cqt_abs, ref=np.max)
    
    # 6. 归一化
    cqt_db = (cqt_db - np.mean(cqt_db)) / np.std(cqt_db)
    
    return cqt_db

def batch_preprocess(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if fname.endswith(".wav"):
            fpath = os.path.join(input_dir, fname)
            cqt_feat = preprocess_audio(fpath)
            np.save(os.path.join(output_dir, fname.replace(".wav", "S")), cqt_feat)
            print(f"Processed {fname}")

if __name__ == "__main__":
    batch_preprocess("data/", "features/")
