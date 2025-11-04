# -*- coding: utf-8 -*-
import os
import numpy as np
import librosa
import librosa.display
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def preprocess_audio(audio_path, sr=48000, duration=2.5, hop_length=512, bins_per_octave=12, n_bins=84):
    """
    对音频进行预处理，输出CQT谱图
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr)

        # 截取或补零
        target_len = int(sr * duration)
        if len(y) > target_len:
            y = y[:target_len]
        else:
            y = np.pad(y, (0, target_len - len(y)), mode='constant')

        # 高频预加重（去低频噪声）
        y = librosa.effects.preemphasis(y)

        # 计算CQT谱图
        cqt = librosa.cqt(
            y=y, sr=sr, hop_length=hop_length,
            bins_per_octave=bins_per_octave, n_bins=n_bins, window='hann'
        )
        cqt_abs = np.abs(cqt)
        cqt_db = librosa.amplitude_to_db(cqt_abs, ref=np.max)

        # 标准化
        cqt_db = (cqt_db - np.mean(cqt_db)) / np.std(cqt_db)
        return cqt_db

    except Exception as e:
        print(f"⚠️ 处理 {audio_path} 失败: {e}")
        return None


def process_one_file(args_tuple):
    """用于多进程的单文件处理函数"""
    fpath, input_dir, output_dir, params = args_tuple
    rel_path = os.path.relpath(fpath, input_dir)
    save_path = os.path.join(output_dir, rel_path.replace(".wav", ".npy"))

    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cqt_feat = preprocess_audio(
        fpath,
        sr=params["sr"],
        duration=params["duration"],
        hop_length=params["hop_length"],
        bins_per_octave=params["bins_per_octave"],
        n_bins=params["n_bins"]
    )

    if cqt_feat is not None:
        np.save(save_path, cqt_feat)
        return fpath
    return None


def batch_preprocess(input_dir, output_dir, sr=48000, duration=2.5,
                     hop_length=512, bins_per_octave=12, n_bins=84, num_workers=4):
    """
    批量预处理音频，支持多层目录和多进程加速
    """
    os.makedirs(output_dir, exist_ok=True)

    # 递归收集所有 wav 文件
    wav_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(".wav"):
                wav_files.append(os.path.join(root, f))
    print(f"🔍 共发现 {len(wav_files)} 个音频文件待处理")

    # 参数打包
    params = dict(sr=sr, duration=duration, hop_length=hop_length,
                  bins_per_octave=bins_per_octave, n_bins=n_bins)

    tasks = [(fpath, input_dir, output_dir, params) for fpath in wav_files]

    # 使用多进程加速
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_one_file, t) for t in tasks]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="特征提取中", ncols=80):
            pass

    print(f"✅ 所有文件已处理完成，特征保存在：{output_dir}")


def get_args():
    parser = argparse.ArgumentParser(description="Real-time audio preprocessing and inference")

    # ----------- 基础路径参数 -----------
    parser.add_argument("--input_dir", type=str, default="data/",
                        help="输入音频文件夹路径（可含多层子目录）")
    parser.add_argument("--output_dir", type=str, default="features/",
                        help="输出特征保存路径（结构与输入一致）")
    parser.add_argument("--audio_path", type=str, default=None,
                        help="单个音频文件路径（可选）")

    # ----------- 特征提取参数 -----------
    parser.add_argument("--sr", type=int, default=48000, help="采样率 (Hz)")
    parser.add_argument("--duration", type=float, default=2.5, help="截取或补零的音频时长（秒）")
    parser.add_argument("--hop_length", type=int, default=512, help="CQT的跳帧长度")
    parser.add_argument("--bins_per_octave", type=int, default=12, help="每个八度的频率数")
    parser.add_argument("--n_bins", type=int, default=84, help="CQT频率总数")

    # ----------- 并行参数 -----------
    parser.add_argument("--num_workers", type=int, default=4, help="并行进程数")
    
    
    # ----------- 实时处理参数 ----------- 
    parser.add_argument("--sr", type=int, default=48000, help="采样率 (Hz)") 
    parser.add_argument("--win_duration", type=float, default=10.0, help="滑窗总长度 (秒)") 
    parser.add_argument("--step_duration", type=float, default=0.5, help="滑窗步长 (秒)") 
    #---------------尚未开发--------------

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # 单文件模式
    if args.audio_path:
        print(f"🎧 处理单个文件: {args.audio_path}")
        feat = preprocess_audio(args.audio_path, sr=args.sr, duration=args.duration,
                                hop_length=args.hop_length, bins_per_octave=args.bins_per_octave,
                                n_bins=args.n_bins)
        if feat is not None:
            save_path = os.path.join(args.output_dir, os.path.basename(args.audio_path).replace(".wav", ".npy"))
            os.makedirs(args.output_dir, exist_ok=True)
            np.save(save_path, feat)
            print(f"✅ 特征已保存至 {save_path}")
    else:
        # 批量模式
        batch_preprocess(args.input_dir, args.output_dir, args.sr, args.duration,
                         args.hop_length, args.bins_per_octave, args.n_bins, args.num_workers)
