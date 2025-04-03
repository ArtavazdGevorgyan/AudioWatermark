# import os
# import torch
# import torchaudio
# import random
# import torchaudio
# from torch.utils.data import Dataset, DataLoader


# def load_audio(file_path, sample_rate=16000):

#     waveform, sr = torchaudio.load(file_path)
#     if sr != sample_rate:
#         waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(
#             waveform
#         )
#     return waveform


# def split_audio(waveform, sample_rate, segment_length, overlap):
#     step = int(segment_length * (1 - overlap) * sample_rate)
#     segment_size = int(segment_length * sample_rate)
#     segments = []
#     for start in range(0, waveform.shape[1] - segment_size + 1, step):
#         segments.append(waveform[:, start : start + segment_size])
#     return segments


# class AudioDataset(Dataset):
#     def __init__(
#         self,
#         data_dir,
#         sample_rate=16000,
#         segment_length=0.1,
#         overlap=0.5,
#         transform=None,
#     ):
#         self.data_dir = data_dir
#         self.sample_rate = sample_rate
#         self.segment_length = segment_length
#         self.overlap = overlap
#         self.transform = transform  # M
#         self.audio_files = [
#             os.path.join(data_dir, f)
#             for f in os.listdir(data_dir)
#             if f.endswith(".wav")
#         ]
#         self.segments = []
#         self.prepare_segments()

#     def prepare_segments(self):
#         for file_path in self.audio_files:
#             waveform = load_audio(file_path, self.sample_rate)
#             self.segments.extend(
#                 split_audio(
#                     waveform, self.sample_rate, self.segment_length, self.overlap
#                 )
#             )

#     def __len__(self):
#         return len(self.segments)

#     def __getitem__(self, idx):
#         waveform = self.segments[idx]

#         if self.transform:
#             waveform = self.transform(waveform)

#         return waveform


# class AudioDataLoader:
#     def __init__(self, data_dir, batch_size=16, shuffle=True, sample_rate=16000):
#         self.dataset = AudioDataset(
#             data_dir, sample_rate, transform=self.default_transform()
#         )
#         print(f"Found {len(self.dataset.audio_files)} audio files in {data_dir}")

#         self.dataloader = DataLoader(
#             self.dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True
#         )

#     def default_transform(self):
#         return torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64)

#     def get_dataloader(self):
#         return self.dataloader


import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import Resample
from split_reconstruct_audio import split_audio, reconstruct_audio


class AudioWatermarkDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000, segment_length=1600, nbits=16):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.nbits = nbits
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith(".wav")]
        self.resample = Resample(orig_freq=sample_rate, new_freq=16000)

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

        # if random.random() < 0.5:
        if False:
            return waveform.squeeze(0), segments, torch.zeros_like(msg), 0.0
        else:
            return waveform.squeeze(0), segments, msg, 1.0


def create_dataloader(data_dir):
    dataset = AudioWatermarkDataset(data_dir)
    # return DataLoader(
    #     dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    # )
    return dataset


# loader = create_dataloader(
#     "/Users/artavazdgevorgyan/Downloads/audioseal_dataset/WAV_format/train_wav",
#     batch_size=16,
# )


# a = AudioWatermarkDataset(
#     "/Users/artavazdgevorgyan/Downloads/audioseal_dataset/WAV_format/train_wav"
# )
# print()
