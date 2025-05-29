import json
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import rcParams


def configure_chinese_support():
    """
    配置 matplotlib 支持中文显示
    """
    rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体显示中文
    rcParams['axes.unicode_minus'] = False   # 解决负号显示为方块的问题


def load_jsonl(file_path):
    """
    读取 JSONL 文件，返回所有行的列表
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def get_data(data, key):
    """
    提取数值和平均值
    """
    try:
        # 提取所有值
        values = [item[key] for item in data if key in item]
        return values
    except KeyError:
        print(f"Key '{key}' not found in one or more items.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []


def plot_hist(wer_data, sim_o_data, utmos_data, output_file):
    """
    绘制直方图在一张图上
    """
    bin_wer_sim = np.linspace(0, 1, 11)  # 为 WER 和 SIM-o 设置区间 (0-0.1, ..., 0.9-1.0)
    bin_utmos = np.linspace(0, 5, 11)     # 为 UTMOS 设置区间 (0-1, ..., 4-5)

    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 一行三列子图

    # WER 直方图
    axes[0].hist(wer_data, bins=bin_wer_sim, color='red', edgecolor='black', alpha=0.4)
    axes[0].set_title("WER直方图", fontsize=28)
    axes[0].set_xlabel("WER", fontsize=24)
    axes[0].set_ylabel("频率", fontsize=24)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    axes[0].tick_params(axis='both', which='major', labelsize=20)

    # SIM-o 直方图
    axes[1].hist(sim_o_data, bins=bin_wer_sim, color='green', edgecolor='black', alpha=0.4)
    axes[1].set_title("SIM-o直方图", fontsize=28)
    axes[1].set_xlabel("SIM-o", fontsize=24)
    axes[1].set_ylabel(" ", fontsize=24)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    axes[1].tick_params(axis='both', which='major', labelsize=20)

    # UTMOS 直方图
    axes[2].hist(utmos_data, bins=bin_utmos, color='skyblue', edgecolor='black', alpha=0.7)
    axes[2].set_title("UTMOS直方图", fontsize=28)
    axes[2].set_xlabel("UTMOS", fontsize=24)
    axes[2].set_ylabel(" ", fontsize=24)
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)
    axes[2].tick_params(axis='both', which='major', labelsize=20)

    # 调整子图布局
    plt.tight_layout()

    # 保存图像为 SVG 格式
    plt.savefig(output_file, format="svg")
    plt.show()
    plt.close()


if __name__ == "__main__":
    langs = ["EN", "ZH", "JP"]
    name = "jp_finetune_jp"
    language = langs[2]
    base_path = r"D:\Desktop\eval_test"

    wer_path = os.path.join(base_path, language, name, "wer.txt")
    sim_o_path = os.path.join(base_path, language, name, "sim-o.jsonl")
    utmos_path = os.path.join(base_path, language, name, "utmos.jsonl")

    configure_chinese_support()

    wer_data = get_data(load_jsonl(wer_path), key="wer")
    sim_o_data = get_data(load_jsonl(sim_o_path), key="SIM-o")
    utmos_data = get_data(load_jsonl(utmos_path), key="utmos")
    print(wer_data, sim_o_data, utmos_data)
    output_file = os.path.join(base_path, language, name, "histogram.svg")
    plot_hist(wer_data, sim_o_data, utmos_data, output_file)

    print(f"Plot has been saved to {output_file}")