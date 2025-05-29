import os
from pathlib import Path
from jiwer import wer
import librosa
import batch_clone_audio
from tqdm import tqdm
import json
import torch.nn.functional as F
import torch
import os.path
from f5_tts.infer.utils_infer import transcribe
from f5_tts.eval.ecapa_tdnn import ECAPA_TDNN_SMALL
import torchaudio
import pypinyin  # 用于中文转拼音
import pykakasi
import prepare_set
import re
from janome.tokenizer import Tokenizer
# 初始化 pykakasi 转换器
kakasi = pykakasi.kakasi()
tokenizer = Tokenizer()

def preprocess_japanese(text):
    """
    使用 Janome 对日语文本进行分词，并转化为平假名。
    """
    # 将文本转换为平假名
    result = kakasi.convert(text)
    hiragana_text = "".join([item['hira'] for item in result])
    # 去除标点符号
    hiragana_text = re.sub(r"[、。]", " ", hiragana_text)

    # 使用 Janome 分词
    tokenized_text = " ".join([token.surface for token in tokenizer.tokenize(hiragana_text)])
    text = " ".join(list(hiragana_text))
    return text

def audio_clone(exp_name, metalst, name=None, ckpt_step = 4000):
    # clone audio and put results in a different folder
    batch_clone_audio.start(seed = 0, exp_name = exp_name, ckpt_step=ckpt_step, test_set=name, metalst = metalst)
    print(f"clone audios is end")

def wer_eval(metalst, gen_wav_dir, output_file, language="en"):
    total_wer = 0
    processed_lines = 0

    with open(metalst, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(output_file, "w", encoding="utf-8") as out_f:
        for line in lines:
            line = line.strip()
            if not line:
                continue  # 跳过空行

            # 假设每行是用逗号分隔，第四个字段是 truth_text
            parts = line.split("\t")  # 如果是空格分隔，改为 line.split()
            if len(parts) < 4:
                print(f"Invalid line format: {line}")
                continue

            filename = parts[0].strip()+".wav"     # 第一列为文件名
            truth_text = parts[3].strip()  # 第四列为 truth_text

            # 获取音频文件路径
            audio_file = os.path.join(gen_wav_dir, filename)

            try:
                # 模拟转录函数，实际替换为具体的 `transcribe` 调用
                hypothesis_text = transcribe(audio_file, language)
                hypothesis_text = hypothesis_text.lower().strip().replace('"', "")
                if language == "zh":
                    # 中文转拼音
                    truth_text = re.sub(r"[,，.。]", "", truth_text)
                    hypothesis_text = re.sub(r"[,，.。]", "", hypothesis_text)
                    truth_text = " ".join(list(truth_text))
                    hypothesis_text = " ".join(list(hypothesis_text))


                elif language == "en":
                    # 英语句首字母大写，其他小写
                    truth_text = truth_text.lower()
                    hypothesis_text = hypothesis_text.lower()
                elif language == "ja":
                    # 日语转平假名
                    truth_text = preprocess_japanese(truth_text)
                    hypothesis_text = preprocess_japanese(hypothesis_text)
                else:
                    raise ValueError(f"Unsupported language: {language}")
                # 计算 WER
                wer_value = wer(truth_text, hypothesis_text)
                total_wer += wer_value
                processed_lines += 1
                json_line = {
                    "filename": filename,
                    "truth_text": truth_text,
                    "hypothesis_text": hypothesis_text,
                    "wer": round(wer_value,4)
                }
                out_f.write(json.dumps(json_line, ensure_ascii=False) + "\n")   # 保存结果到文件
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                continue

        # 计算平均 WER
        average_wer = total_wer / processed_lines if processed_lines > 0 else 0
        avg_json_line = {
            "Average WER": round(average_wer, 4)
        }
        out_f.write(json.dumps(avg_json_line, ensure_ascii=False) + "\n")

    print(f"Processing complete. Results saved to {output_file}")

def SIM_o_eval(metalst, gen_wav_dir, output_file):
    # 套用 sim-o-eval
    device = f"cuda:{0}"
    ckpt_dir = r"E:\github\F5-TTS\ckpts\wavlm_large_finetune.pth"
    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
    state_dict = torch.load(ckpt_dir, weights_only=True, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict["model"], strict=False)

    use_gpu = True if torch.cuda.is_available() else False
    if use_gpu:
        model = model.cuda(device)
    model.eval()

    total_sim = 0
    processed_lines = 0

    with open(metalst, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(output_file, "w", encoding="utf-8") as out_f:
        for line in lines:
            line = line.strip()
            if not line:
                continue  # 跳过空行

            # 假设每行是用逗号分隔，第四个字段是 truth_text
            parts = line.split("\t")  # 如果是空格分隔，改为 line.split()
            if len(parts) < 4:
                print(f"Invalid line format: {line}")
                continue

            filename = parts[0].strip()         # 第一列为文件名
            file = parts[0].strip()+".wav"
            prompt_wav = parts[4].strip()  # 第四列为 truth_text

            # 获取音频文件路径
            gen_wav = os.path.join(gen_wav_dir, file)

            try:
                wav1, sr1 = torchaudio.load(gen_wav)
                wav2, sr2 = torchaudio.load(prompt_wav)

                resample1 = torchaudio.transforms.Resample(orig_freq=sr1, new_freq=16000)
                resample2 = torchaudio.transforms.Resample(orig_freq=sr2, new_freq=16000)
                wav1 = resample1(wav1)
                wav2 = resample2(wav2)

                if use_gpu:
                    wav1 = wav1.cuda(device)
                    wav2 = wav2.cuda(device)
                with torch.no_grad():
                    emb1 = model(wav1)
                    emb2 = model(wav2)

                sim = F.cosine_similarity(emb1, emb2)[0].item()
                total_sim += sim
                processed_lines += 1


                json_line = {
                    "filename": filename,
                    "gen_wav": Path(gen_wav).stem,
                    "prompt_path": Path(prompt_wav).stem,
                    "SIM-o": round(sim, 4),
                }
                out_f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
                # print(f"VSim score between two audios: {sim:.4f} (-1.0, 1.0).")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

        # 计算平均 SIM-o
        average_sim = total_sim / processed_lines if processed_lines > 0 else 0
        avg_json_line = {
            "Average SIM-o": round(average_sim, 4)
        }
        out_f.write(json.dumps(avg_json_line, ensure_ascii=False) + "\n")

    print(f"Processing complete. Results saved to {output_file}")

def utmos_eval(audio_dir, ext="wav", utmos_result_path=None):
    # UTMOS Evaluation  貌似只适合用于english test
    if utmos_result_path is None:
        utmos_result_path = Path(audio_dir) / "_utmos_results.jsonl"
    device = "cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu"

    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    predictor = predictor.to(device)

    audio_paths = list(Path(audio_dir).rglob(f"*.{ext}"))
    utmos_score = 0

    with open(utmos_result_path, "w", encoding="utf-8") as f:
        for audio_path in tqdm(audio_paths, desc="Processing"):
            wav, sr = librosa.load(audio_path, sr=None, mono=True)
            wav_tensor = torch.from_numpy(wav).to(device).unsqueeze(0)
            score = predictor(wav_tensor, sr)
            line = {}
            line["wav"], line["utmos"] = str(audio_path.stem), score.item()
            utmos_score += score.item()
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        avg_score = utmos_score / len(audio_paths) if len(audio_paths) > 0 else 0
        avg_json_line = {
            "Average UTMOS": round(avg_score, 4)
        }
        f.write(json.dumps(avg_json_line, ensure_ascii=False) + "\n")


    print(f"UTMOS: {avg_score:.4f}")
    print(f"UTMOS results saved to {utmos_result_path}")

def metalst_modify(metalst, test_nums, reproduce = False):

    if not os.path.exists(metalst):
        raise FileNotFoundError(f"文件 {metalst} 不存在。")

    # 构造新文件的路径
    dir_name, file_name = os.path.split(metalst)
    testlst_path = os.path.join(dir_name, "testlst.lst")
    # 检查文件是否存在
    if os.path.exists(testlst_path) and reproduce == False:
        return testlst_path

    # 读取文件内容
    with open(metalst, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 检查文件是否有足够的行
    if len(lines) < test_nums:
        raise ValueError(f"文件 {metalst} 中的行数不足 {test_nums} 行。")

    # 提取前 test_nums 行
    test_lines = lines[:test_nums]

    # 写入新文件
    with open(testlst_path, "w", encoding="utf-8") as f:
        f.writelines(test_lines)

    return testlst_path


if __name__ == "__main__":
    # language choice
    langs = ["en", "zh", "ja"]
    language = langs[2]
    exp_name = "jp_base"   #实验名称，模型选自其中的步长
    ckpt_step = 180000
    name = "jp_finetune_jp"
    test_number = 20
    gen_wav_dir = rf"E:\github\F5-TTS\results\{exp_name}_{ckpt_step}\{name}\seed0_euler_nfe32_vocos_ss-1_cfg2.0_speed1.0"
    # language choice
    state = True
    if language == "en":
        test_path = r"D:\Desktop\eval_test\EN\dataset"
        if not os.path.exists(os.path.join(test_path, "../", name)):
            os.makedirs(os.path.join(test_path, "../", name))
        metalst = os.path.join(test_path,"../",  name, "metalst.lst")
        wer_file = os.path.join(test_path,"../", name, "wer.txt")
        sim_o_file = os.path.join(test_path, "../", name, "SIM-o.jsonl")
        utmos_file = os.path.join(test_path, "../", name, "UTMOS.jsonl")
    elif language == "zh":
        test_path = r"D:\Desktop\eval_test\ZH\dataset"
        if not os.path.exists(os.path.join(test_path, "../", name)):
            os.makedirs(os.path.join(test_path, "../", name))
        metalst = os.path.join(test_path, "../", name, "metalst.lst")
        wer_file = os.path.join(test_path, "../", name, "wer.txt")
        sim_o_file = os.path.join(test_path, "../", name, "SIM-o.jsonl")
        utmos_file = os.path.join(test_path, "../", name, "UTMOS.jsonl")
    elif language == "ja":
        test_path = r"D:\Desktop\eval_test\JP\dataset"
        if not os.path.exists(os.path.join(test_path, "../", name)):
            os.makedirs(os.path.join(test_path, "../", name))
        metalst = os.path.join(test_path, "../", name, "metalst.lst")
        wer_file = os.path.join(test_path, "../", name, "wer.txt")
        sim_o_file = os.path.join(test_path, "../", name, "SIM-o.jsonl")
        utmos_file = os.path.join(test_path, "../", name, "UTMOS.jsonl")
    else:
        print("Language not supported.")
        state = False
    # step1: clone 语音
    if state:
        # step0:
        # prepare_set.start(test_path, name)    # 创建csv和metalst文件
        #
        # step1:  clone 语音
        metalst = metalst_modify(metalst, test_nums = test_number, reproduce = False)
        audio_clone(exp_name, metalst, name=name, ckpt_step = ckpt_step)
        #
        # # step2: wer 计算
        wer_eval(metalst, gen_wav_dir, wer_file, language=language)
        #
        # step3: SIM-o计算
        SIM_o_eval(metalst, gen_wav_dir, sim_o_file)

        # step4:  utmos 计算
        utmos_eval(gen_wav_dir, ext="wav", utmos_result_path=utmos_file)
    else:
        print(r"check your language, must be in 'en', 'zh', 'jp'.")