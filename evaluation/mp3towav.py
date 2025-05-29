import os
import librosa
from scipy.io import wavfile
from tqdm import tqdm
from glob import glob
import numpy as np
from f5_tts.infer.utils_infer import transcribe

# 主处理逻辑
def process_mp3_dataset(path_dataset, output_file, language="en"):
    # 搜索 MP3 文件
    file_audios = glob(os.path.join(path_dataset, "*.mp3"))
    if not file_audios:
        print("No MP3 files were found in the dataset.")
        return

    num = 0
    error_num = 0
    data = ""

    # 处理每个 MP3 文件
    for file_audio in tqdm(file_audios, desc="Processing MP3 files", total=len(file_audios)):
        try:
            # 加载音频
            audio, _ = librosa.load(file_audio, sr=24000, mono=True)

            # 保存为 WAV 格式（可选）
            file_segment = os.path.splitext(file_audio)[0] + ".wav"
            wavfile.write(file_segment, 24000, (audio * 32767).astype(np.int16))

            # 转录
            try:
                text = transcribe(file_segment, language)
                text = text.lower().strip().replace('"', "")
                data += f"{os.path.basename(file_segment)}|{text}\n"
                num += 1
            except Exception as e:
                print(f"Error transcribing {file_segment}: {e}")
                error_num += 1
        except Exception as e:
            print(f"Error processing {file_audio}: {e}")
            error_num += 1

    # 保存结果到输出文件
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(data)

    print(f"Processing complete. Total files processed: {num}, Errors: {error_num}")


if __name__ == "__main__":
    # 输入路径和输出路径
    dataset_path = "D:\Desktop\eval_test\en_test2\data"
    output_transcription_file = r"D:\Desktop\eval_test\en_test2\transcriptions.txt"

    # 运行处理逻辑
    process_mp3_dataset(dataset_path, output_transcription_file, language="en")