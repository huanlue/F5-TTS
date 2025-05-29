import os
import json
import csv
from collections import defaultdict


def process_json_files(folder_path, output_main_csv, output_remaining_csv):
    # 用于存储每个 speaker 的数据
    speaker_data = defaultdict(list)

    # 遍历文件夹中的文件
    for file_name in os.listdir(folder_path):
        # 如果是 JSON 文件
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            try:
                # 读取 JSON 文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    # 筛选 duration 在 3 到 10 秒之间的记录
                    duration = data["duration"]
                    if 2 <= duration <= 10:
                        speaker = data["speaker"]
                        file_name_with_wav = data["id"] + ".mp3"
                        speaker_data[speaker].append({
                            "id_with_wav": file_name_with_wav,
                            "text": data["text"]
                        })
                    else:
                        print(f"{file_path}的duration:{duration}不满足条件。")
            except json.JSONDecodeError:
                print(f"无法解析 JSON 文件：{file_name}")
            except KeyError as e:
                print(f"JSON 文件缺少必要字段：{file_name}, 缺少 {e}")
            except Exception as e:
                print(f"处理文件 {file_name} 时出错：{e}")

    # 筛选出至少有两个语音记录的 speaker
    filtered_speaker_data = {
        speaker: records for speaker, records in speaker_data.items() if len(records) >= 2
    }

    # 写入主 CSV 和剩余 CSV
    try:
        with open(output_main_csv, 'w', newline='', encoding='utf-8') as main_csv, \
                open(output_remaining_csv, 'w', newline='', encoding='utf-8') as remaining_csv:

            main_writer = csv.writer(main_csv)
            remaining_writer = csv.writer(remaining_csv)

            # 写入 CSV 文件的表头
            main_writer.writerow(["speaker", "id_with_wav", "text"])
            remaining_writer.writerow(["speaker", "id_with_wav", "text"])

            # 遍历每个 speaker 的数据
            for speaker, records in filtered_speaker_data.items():
                # 主 CSV 文件中的第一条记录
                main_writer.writerow([speaker, records[0]["id_with_wav"], records[0]["text"]])

                # 剩余记录写入剩余 CSV 文件
                for record in records[1:]:
                    remaining_writer.writerow([speaker, record["id_with_wav"], record["text"]])

        print(f"数据已成功写入到 {output_main_csv} 和 {output_remaining_csv}")
    except Exception as e:
        print(f"写入 CSV 文件时出错：{e}")


def generate_metalist(main_csv, remaining_csv, output_file, base_path):
    """
    生成 metalist 文件
    """
    try:
        with open(main_csv, 'r', encoding='utf-8') as main_file, \
                open(remaining_csv, 'r', encoding='utf-8') as remaining_file:

            main_reader = csv.DictReader(main_file)

            # 打开输出文件
            with open(output_file, 'w', encoding='utf-8') as output:
                # 写入 metalist 文件
                for main_row in main_reader:
                    speaker = main_row["speaker"]
                    ref_txt = main_row['text']
                    ref_wav = os.path.join(base_path, main_row["id_with_wav"])
                    num = 1

                    # 重新初始化 remaining_reader
                    remaining_file.seek(0)  # 将 remaining_file 指针重置到文件开头
                    remaining_reader = csv.DictReader(remaining_file)

                    # 遍历剩余 CSV 文件中的记录
                    for remaining_row in remaining_reader:
                        if num < 4:
                            if remaining_row["speaker"] == speaker:
                                gen_txt = remaining_row["text"]
                                gen_wav = os.path.join(base_path, remaining_row["id_with_wav"])

                                # 写入一行 metalist 数据
                                output.write(f"{speaker}_{num}\t{ref_txt}\t{ref_wav}\t{gen_txt}\t{gen_wav}\n")
                                print(f"{speaker}\t{ref_txt}\t{ref_wav}\t{gen_txt}\t{gen_wav}\n")
                                num += 1
                        else:
                            break  # 提前跳出循环，不再检查剩余行
                print(f"metalist 文件已成功生成：{output_file}")

    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
    except KeyError as e:
        print(f"CSV 文件缺少必要字段: {e}")
    except Exception as e:
        print(f"生成 metalist 文件时出错: {e}")

def csv_create(folder, name):

    if os.path.isdir(folder):
        # 输出的 CSV 文件
        ref_path = os.path.join(folder, "../", name, "ref.csv")
        gen_path = os.path.join(folder, "../", name, "gen.csv")

        # 处理 JSON 文件并生成 CSV
        process_json_files(folder, ref_path, gen_path)
    else:
        print("输入的路径不是一个有效的文件夹。")

def metalst_create(folder, name):
    # 输入主 CSV 文件和剩余 CSV 文件路径
    if os.path.isdir(folder):
        ref_path = os.path.join(folder, "../", name, "ref.csv")
        gen_path = os.path.join(folder, "../", name, "gen.csv")
    else:
        print("输入的路径不是一个有效的文件夹。")
        return -1
    # 输出文件名
    metalst_file = os.path.join(folder,"../", name, "metalst.lst")
    # 生成 metalist 文件
    generate_metalist(ref_path, gen_path, metalst_file, folder)


def start(folder, name):
    csv_create(folder, name)
    metalst_create(folder, name)