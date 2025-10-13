# visualize_cqt.py
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

def visualize_waveform_and_cqt(npy_path, audio_path=None, save_dir=None,
                               sr=48000, hop_length=512, bins_per_octave=12):
    """
    同时显示：
    1️⃣ 原始音频波形
    2️⃣ CQT 频谱图
    并可保存为 .png 文件
    """

    # === 1. 加载CQT数据 ===
    cqt_data = np.load(npy_path)
    print(f"Loaded CQT from {npy_path}, shape={cqt_data.shape}")

    # === 2. 如果提供了音频路径，加载音频 ===
    y = None
    if audio_path and os.path.exists(audio_path):
        y, sr = librosa.load(audio_path, sr=sr)
        print(f"Loaded audio: {audio_path}, length={len(y)/sr:.2f}s, sr={sr}")
    else:
        print("⚠️ 未提供音频文件，只显示谱图。")

    # === 3. 创建图像布局 ===
    fig, axs = plt.subplots(2 if y is not None else 1, 1,
                            figsize=(10, 6 if y is not None else 4),
                            sharex=False)

    # === 4. 绘制波形图 ===
    if y is not None:
        librosa.display.waveshow(y, sr=sr, ax=axs[0], color='gray')
        axs[0].set_title("Waveform")
        axs[0].set_ylabel("Amplitude")
        axs[0].set_xlabel("Time (s)")

    # === 5. 绘制CQT谱图 ===
    ax_spec = axs[1] if y is not None else axs
    img = librosa.display.specshow(
        cqt_data,
        sr=sr,
        hop_length=hop_length,
        bins_per_octave=bins_per_octave,
        x_axis='time',
        y_axis='cqt_note',
        ax=ax_spec
    )
    ax_spec.set_title("CQT Spectrogram")
    fig.colorbar(img, ax=ax_spec, format="%+2.0f dB")

    plt.tight_layout()

    # === 6. 保存 ===
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, os.path.basename(npy_path).replace(".npy", ".png"))
        plt.savefig(save_path, dpi=200)
        print(f"Saved combined visualization to {save_path}")

    plt.show()


if __name__ == "__main__":
    # 默认参数
    feature_dir = "features/"
    audio_dir = "data/"
    save_dir = "spectrograms/"

    # 自动选取第一个文件
    npy_files = [f for f in os.listdir(feature_dir) if f.endswith(".npy")]
    if not npy_files:
        print("⚠️ 没有找到 .npy 文件，请先运行 preprocess_snore.py")
    else:
        fname = npy_files[0]
        npy_path = os.path.join(feature_dir, fname)
        audio_path = os.path.join(audio_dir, fname.replace(".npy", ".wav"))

        visualize_waveform_and_cqt(npy_path, audio_path, save_dir=save_dir)
