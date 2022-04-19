# reference
# GitHub@xcmyz: https://github.com/xcmyz/ConvTasNet4BasisMelGAN/preprocess.py

# modified and re-distributed by Zifeng Zhao @ Peking University

import torch

import os
import audio
import random
import numpy as np
import hparams as hp

from tqdm import tqdm
from functools import partial
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, type=str, help='dataset name')
parser.add_argument('--dataset_dir', required=True, type=str, help='dataset directory')
parser.add_argument('--save_dir', required=True, type=str, help='path to save training data (.npy files)')
args = parser.parse_args()

DATASET = args.dataset

def _process_utterance(in_path, out_path, index):
    wav = torch.Tensor(audio.load_wav(in_path, hp.sample_rate, encode=False))
    noi = audio.add_noise(wav, quantization_channel=hp.quantization_channel)
    mix = wav.float() + noi

    wav_name = f"{index}.wav.npy"
    noi_name = f"{index}.noi.npy"
    mix_name = f"{index}.mix.npy"

    np.save(os.path.join(args.save_dir, wav_name), wav.numpy(), allow_pickle=False)
    np.save(os.path.join(args.save_dir, noi_name), noi.numpy(), allow_pickle=False)
    np.save(os.path.join(args.save_dir, mix_name), mix.numpy(), allow_pickle=False)


def get_pathfile():
    if DATASET == "biaobei":
        with open("BZNSYP.txt", "w", encoding="utf-8") as f:
            for filename in os.listdir(os.path.join(args.dataset_dir, "Wave")):
                if filename[0] != ".":
                    f.write(os.path.abspath(os.path.join(args.dataset_dir, "Wave", filename)) + "\n")
    elif DATASET == "aishell3":
        cnt = 0
        with open("aishell3.txt", "w", encoding="utf-8") as f:
            files = []
            wav_path = os.path.join("data_aishell3", "train", "wav")
            for speaker_name in os.listdir(wav_path):
                path = os.path.join(wav_path, speaker_name)
                for wav_name in os.listdir(path):
                    cnt += 1
                    files.append(os.path.abspath(os.path.join(path, wav_name)))
            wav_path = os.path.join("data_aishell3", "test", "wav")
            for speaker_name in os.listdir(wav_path):
                path = os.path.join(wav_path, speaker_name)
                for wav_name in os.listdir(path):
                    cnt += 1
                    files.append(os.path.abspath(os.path.join(path, wav_name)))
            print(f"load {cnt} files.")
            files = random.sample(files, hp.dataset_size)
            for file in files:
                f.write(f"{file}\n")
    else:
        with open("dataset.txt", "w", encoding="utf-8") as f:
            for filename in os.listdir(os.path.join("largedata")):
                filepath = os.path.join("largedata", filename)
                f.write(f"{filepath}\n")


if __name__ == "__main__":
    # Get path in a directory
    get_pathfile()
    
    executor = ProcessPoolExecutor(max_workers=cpu_count() - 1)
    futures = []
    os.makedirs(hp.dataset_path, exist_ok=True)
    if DATASET == "biaobei":
        filename = "BZNSYP.txt"
    elif DATASET == "aishell3":
        filename = "aishell3.txt"
    else:
        filename = "dataset.txt"
    with open(filename, "r", encoding="utf-8") as f:
        paths = f.readlines()
        length = len(paths)
        for i in tqdm(range(length)):
            path = paths[i]
            path = path[:-1]
            index = path.split("/")[-1]
            futures.append(executor.submit(partial(_process_utterance, os.path.join(path), hp.dataset_path, index)))
    [future.result() for future in tqdm(futures)]
