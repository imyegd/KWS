import os
import json
import wave
import numpy as np
import wave
import numpy as np
from python_speech_features import mfcc
import scipy
import torch
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from wave import open as wave_open
import pandas as pd

def get_labels(audio_paths):
    labels = []
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    
    for path in audio_paths:
        word = path.split('\\')[0]
        labels.append(config["label"][word])
    
    return labels


def get_path(mode):
    
    with open('data/testing_list.txt', 'r', encoding='utf-8') as f:
        test_path = f.readlines()
        test_path = [element.strip() for element in test_path if element]

    with open('data/validating_list.txt', 'r', encoding='utf-8') as f:
        val_path = f.readlines()
        val_path = [element.strip() for element in val_path if element]

    with open('data/training_list.txt', 'r', encoding='utf-8') as f:
        train_path = f.readlines()
        train_path = [element.strip() for element in train_path if element]

    # train_path = []
    # for root, dirs, files in os.walk('data/data_mini_merge'):
    #     for filename in files:
    #         if filename.endswith('.wav'):
    #             # 打印文件的完整路径
    #             path = os.path.join(root, filename)
    #             path = path.split('data_mini_merge\\')[-1]
    #             train_path.append(path)
    # with open('data/training_list.txt', 'w', encoding='utf-8') as f:
    #     f.write('\n'.join(train_path))
    if mode == 'test':
        return test_path
    elif mode == 'val':
        return val_path
    elif mode == 'train':
        return train_path
    

# 归一化函数
def normalize(wav_data):
    max_val = np.max(np.abs(wav_data))
    if max_val > 0:
        return wav_data / max_val
    return wav_data

# 长度标准化函数 - 这里我们选择填充（扩充）方式
def standardize_length(wav_data, desired_length):
    current_length = len(wav_data)
    if current_length == desired_length:
        return wav_data
    elif current_length < desired_length:
        padding = np.zeros(desired_length - current_length)
        return np.concatenate((wav_data, padding))
    else:  # 当前长度大于期望长度时进行截断
        return wav_data[:desired_length]

# 读取WAV文件并处理
def process_wav_files(wav_file_name, desired_length):

    # 打开WAV文件
    with wave.open(wav_file_name, 'rb') as wav_file:
        # 读取音频数据
        wav_data = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)
        
        # 归一化音频数据
        wav_data_normalized = normalize(wav_data)
        
        # 长度标准化
        wav_data_standardized = standardize_length(wav_data_normalized, desired_length)

            
        # 读取波形数据
        frames = wav_file.readframes(-1)
        sample_rate = wav_file.getframerate()


        # 提取 MFCCs
        # mfccs = mfcc(wav_data_standardized, samplerate=sample_rate)
        numcep = 13  # MFCC 系数的数量
        nfft = 512  # FFT 的窗口大小
        winlen = 0.025  # 窗口长度（秒）
        winstep = 0.01  # 窗口步长（秒）
        highfreq = 0.5 * sample_rate  # 最高频率截止值

        # 计算 MFCCs
        # 注意：确保 winlen 和 winstep 乘以 sample_rate 后的结果能整除音频信号的长度
        mfccs = mfcc(
            wav_data_standardized,
            samplerate=sample_rate,
            nfft=nfft,
            winlen=winlen,
            winstep=winstep,
            numcep=numcep,
            highfreq=highfreq
        )
        # print(mfccs.shape)
        return mfccs


def getloader(mode):
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)


    path = get_path(mode)
    label = get_labels(path)
    dataset = WavDataset(path, label, config)
    loader = DataLoader(dataset, batch_size=config["batchsize"], shuffle=True)
    return loader


class WavDataset(Dataset):
    def __init__(self, audio_paths, labels, config, transform=None, desired_length=16000):
        self.config = config
        self.audio_paths = audio_paths
        self.labels = pd.get_dummies(labels, dummy_na=False).reindex(columns=self.config["label"].values(), fill_value=False)
        self.labels = torch.tensor(np.array(self.labels.astype(int)))
        self.transform = transform
        self.desired_length = desired_length
        

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        # 加载WAV文件
        file_path = os.path.join(self.config["data_path"], self.audio_paths[idx])
        wav_data = process_wav_files(file_path, self.desired_length)
        wav_data = wav_data.astype(np.float32)
        # 应用transform
        if self.transform:
            wav_data = self.transform(wav_data)

        # 标签
        # label = self.labels[idx].to(torch.float32) if self.labels is not None else None
        label = self.labels[idx].float()

        if wav_data.shape[0] != 99:
            print(self.audio_paths[idx])

        # wav_data = torch.from_numpy(wav_data).float()
        
        return wav_data, label


if __name__=="__main__":
    # 假设你已经有了WAV文件路径列表和对应的标签列表
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    path = get_path('train')
    label = get_labels(path)

    # 创建数据集实例
    dataset = WavDataset(path, label, config)

    dataloader = DataLoader(dataset, batch_size=16)

    i = 0
    for batch in dataloader:
        print(123 * 16)
        i += 1
        print(batch[0].size())

    # process_wav_files('data\data_mini_merge\Silence\self_audio.wav_0.wav', 16000)
    