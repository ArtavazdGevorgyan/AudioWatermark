import torch
import torchaudio


def split_audio(waveform, sample_rate, segment_length_ms=100, overlap=0.5):
    segment_length = int((segment_length_ms / 1000) * sample_rate)
    step = int(segment_length * (1 - overlap))

    if waveform.shape[1] < segment_length:
        print("Audio too short, returning empty tensor.")
        return torch.empty(0, 1, segment_length)  # Return an empty 3D tensor

    # Extract overlapping segments using `unfold`
    segments = waveform.unfold(1, segment_length, step).squeeze(
        0
    )  # (num_segments, segment_length)

    # Add a new dimension to match (num_segments, 1, segment_length)
    return segments.unsqueeze(1)


audio_path = "/Users/artavazdgevorgyan/Desktop/segment_000.wav"
waveform, sr = torchaudio.load(audio_path)
splitted_audio = split_audio(waveform, sr)
# print(splitted_audio)


def reconstruct_audio(segments, sample_rate, overlap=0.5):
    num_segments, _, segment_length = segments.shape
    step = int(segment_length * (1 - overlap))
    total_length = (num_segments - 1) * step + segment_length

    waveform = torch.zeros(1, total_length)
    waveform = waveform.to(segments.device)  # Move to the same device as segments
    # Hann window for smooth overlap
    window = torch.hann_window(segment_length).unsqueeze(
        0
    )  # Shape: (1, segment_length)
    window = window.to(segments.device)  # Move window to the same device as segments
    for i in range(num_segments):
        start = i * step
        end = start + segment_length
        waveform[:, start:end] += segments[i] * window  # Apply windowing while summing

    return waveform


# rec_wav = reconstruct_audio(splitted_audio, sr)
# print(waveform.shape)
# print(rec_wav.shape)
# print(rec_wav)
# torchaudio.save("/Users/artavazdgevorgyan/Desktop/segment_0.wav", rec_wav, 16_000)
# # print(sum(waveform - rec_wav)))
# print(sum(sum(waveform - rec_wav)))
