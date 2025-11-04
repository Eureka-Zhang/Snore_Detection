from pydub import AudioSegment
import os

def convert_mp3_to_wav(input_folder, output_folder):
    """
    批量将指定文件夹中的 mp3 文件转换为 wav 格式。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".mp3"):
            mp3_path = os.path.join(input_folder, filename)
            wav_filename = os.path.splitext(filename)[0] + ".wav"
            wav_path = os.path.join(output_folder, wav_filename)

            print(f"正在转换: {filename} → {wav_filename}")
            sound = AudioSegment.from_mp3(mp3_path)
            sound.export(wav_path, format="wav")
    
    print("✅ 全部转换完成！")

if __name__ == "__main__":
    input_dir = "mp3_files"   # 输入文件夹
    output_dir = "wav_files"  # 输出文件夹
    convert_mp3_to_wav(input_dir, output_dir)
