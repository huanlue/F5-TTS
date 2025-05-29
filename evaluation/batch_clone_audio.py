import os
import sys

sys.path.append(os.getcwd())

import argparse
import time
from importlib.resources import files

import torch
import torchaudio
from accelerate import Accelerator
from hydra.utils import get_class
from omegaconf import OmegaConf
from tqdm import tqdm

from f5_tts.eval.utils_eval import (
    get_inference_prompt,
    get_librispeech_test_clean_metainfo,
    get_seedtts_testset_metainfo,
)
from f5_tts.infer.utils_infer import load_checkpoint, load_vocoder
from f5_tts.model import CFM
from f5_tts.model.utils import get_tokenizer


accelerator = Accelerator()
device = f"cuda:{accelerator.process_index}"

use_ema = True
target_rms = 0.1

rel_path = r"E:/github/F5-TTS/"


def get_metainfo(path):
    metainfo = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):  # 使用 enumerate 获取行号
            try:
                # 将每一行分割为所需的5个字段
                gen_utt, ref_txt, ref_wav, gen_txt, gen_wav = line.strip().split("\t")
                # 将结果存入元组并添加到列表
                metainfo.append((gen_utt, ref_txt, ref_wav, gen_txt, gen_wav))
            except ValueError as e:  # 捕获分割字段不足的异常
                print(f"Error in line {i + 1}: {line.strip()}")  # 打印错误行号和内容
                print(f"Exception: {e}")  # 打印具体的异常原因
                continue
    return metainfo
def start(seed = 0, exp_name = "en_test", ckpt_step=4000, test_set=None, metalst = None ):
    # test data set
    if test_set is None:
        print("testset is unset. Check it and retry.")
        return

    nfe_step = 32   #int  16, 32
    ode_method = "euler"
    sway_sampling_coef = -1

    # testset = args.testset

    infer_batch_size = 1  # max frames. 1 for ddp single inference (recommended)
    cfg_strength = 2.0
    speed = 1.0
    use_truth_duration = False
    no_ref_audio = False

    model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{exp_name}.yaml")))
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch

    dataset_name = model_cfg.datasets.name
    tokenizer = model_cfg.model.tokenizer

    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
    target_sample_rate = model_cfg.model.mel_spec.target_sample_rate
    n_mel_channels = model_cfg.model.mel_spec.n_mel_channels
    hop_length = model_cfg.model.mel_spec.hop_length
    win_length = model_cfg.model.mel_spec.win_length
    n_fft = model_cfg.model.mel_spec.n_fft

#   testset_loading
#     testset = "librispeech"
#     metalst = rel_path + "/data/librispeech_pc_test_clean_cross_sentence.lst"
#     librispeech_test_clean_path = r"E:/github/F5-TTS/evaluation/test-clean/"  # test-clean path

    metainfo = get_metainfo(metalst)

    r'''    gen_utt, ref_txt, ref_wav, " " + gen_txt, gen_wav:
    (('4992-23283-0000', 'exclaimed Bill Harmon to his wife as they went through the lighted hall.', 
    'E:/github/F5-TTS/evaluation/test-clean/4992\\41806\\4992-41806-0009.flac', 
    ' But the more forgetfulness had then prevailed, the more powerful was the force of remembrance when she awoke.',
    'E:/github/F5-TTS/evaluation/test-clean/4992\\23283\\4992-23283-0000.flac')
    '''

    # path to save genereted wavs
    output_dir = (
        f"{rel_path}/"
        f"results/{exp_name}_{ckpt_step}/{test_set}/"
        f"seed{seed}_{ode_method}_nfe{nfe_step}_{mel_spec_type}"
        f"{f'_ss{sway_sampling_coef}' if sway_sampling_coef else ''}"
        f"_cfg{cfg_strength}_speed{speed}"
        f"{'_gt-dur' if use_truth_duration else ''}"
        f"{'_no-ref-audio' if no_ref_audio else ''}"
    )
    # # 保存metainfo到metainfo.txt
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # with open("D:/desktop" + "/metainfo.txt", "w", encoding="utf-8") as f:
    #     f.write(str(metainfo))
    #     f.close()
    # print("metainfo.txt saved to ", "D:/desktop" + "/metainfo.txt")
    # -------------------------------------------------#

    prompts_all = get_inference_prompt(
        metainfo,
        speed=speed,
        tokenizer=tokenizer,
        target_sample_rate=target_sample_rate,
        n_mel_channels=n_mel_channels,
        hop_length=hop_length,
        mel_spec_type=mel_spec_type,
        target_rms=target_rms,
        use_truth_duration=use_truth_duration,
        infer_batch_size=infer_batch_size,
    )

    # Vocoder model
    local = False
    if mel_spec_type == "vocos":
        vocoder_local_path = "../checkpoints/charactr/vocos-mel-24khz"
    elif mel_spec_type == "bigvgan":
        vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"
    vocoder = load_vocoder(vocoder_name=mel_spec_type, is_local=local, local_path=vocoder_local_path)

    # Tokenizer
    vocab_char_map, vocab_size = get_tokenizer(dataset_name, tokenizer)

    # Model
    model = CFM(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)


    ckpt_path = rel_path + f"/ckpts/{exp_name}/model_{ckpt_step}.pt"
    if not os.path.exists(ckpt_path):
        try:
            ckpt_path = rel_path + f"/ckpts/{exp_name}/model_{ckpt_step}.safetensors"
        except:
            raise("Loading from self-organized training checkpoints rather than released pretrained.")
        # ckpt_path = rel_path + f"/{model_cfg.ckpts.save_dir}/model_{ckpt_step}.pt"
    dtype = torch.float32 if mel_spec_type == "bigvgan" else None
    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

    if not os.path.exists(output_dir) and accelerator.is_main_process:
        os.makedirs(output_dir)

    # start batch inference
    accelerator.wait_for_everyone()
    start = time.time()

    with accelerator.split_between_processes(prompts_all) as prompts:
        for prompt in tqdm(prompts, disable=not accelerator.is_local_main_process):
            utts, ref_rms_list, ref_mels, ref_mel_lens, total_mel_lens, final_text_list = prompt
            ref_mels = ref_mels.to(device)
            ref_mel_lens = torch.tensor(ref_mel_lens, dtype=torch.long).to(device)
            total_mel_lens = torch.tensor(total_mel_lens, dtype=torch.long).to(device)

            # Inference
            with torch.inference_mode():
                generated, _ = model.sample(
                    cond=ref_mels,
                    text=final_text_list,
                    duration=total_mel_lens,
                    lens=ref_mel_lens,
                    steps=nfe_step,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                    no_ref_audio=no_ref_audio,
                    seed=seed,
                )
                # Final result
                for i, gen in enumerate(generated):
                    gen = gen[ref_mel_lens[i] : total_mel_lens[i], :].unsqueeze(0)
                    gen_mel_spec = gen.permute(0, 2, 1).to(torch.float32)
                    if mel_spec_type == "vocos":
                        generated_wave = vocoder.decode(gen_mel_spec).cpu()
                    elif mel_spec_type == "bigvgan":
                        generated_wave = vocoder(gen_mel_spec).squeeze(0).cpu()

                    if ref_rms_list[i] < target_rms:
                        generated_wave = generated_wave * ref_rms_list[i] / target_rms
                    torchaudio.save(f"{output_dir}/{utts[i]}.wav", generated_wave, target_sample_rate)
                    print("generated auido has saved to path:", f"{output_dir}/{utts[i]}.wav")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        timediff = time.time() - start
        print(f"Done batch inference in {timediff / 60:.2f} minutes.")
