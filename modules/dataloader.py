import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import Resample
from split_reconstruct_audio import split_audio


class AudioWatermarkDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000, segment_length=1600, nbits=16):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.nbits = nbits
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith(".wav")]
        self.resample = Resample(orig_freq=sample_rate, new_freq=16_000)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_idx = idx % len(self.file_list)
        audio_path = os.path.join(self.root_dir, self.file_list[file_idx])
        waveform, sr = torchaudio.load(audio_path)

        if sr != self.sample_rate:
            waveform = self.resample(waveform)

        # Split audio into segments
        segments = split_audio(waveform, self.sample_rate)

        # Generate random message
        msg = torch.randint(0, 2, (self.nbits,)).float()
        msg = msg.repeat(len(segments), 1)

        # 50% chance to return non-watermarked
        if random.random() < 0.5:
            return waveform.squeeze(0), segments, torch.zeros_like(msg), 0.0
        else:
            return waveform.squeeze(0), segments, msg, 1.0


def create_dataloader(data_dir):
    dataset = AudioWatermarkDataset(data_dir)
    return dataset
