import torch


def split_audio(waveform, sample_rate, segment_length_ms=100, overlap=0.5):
    segment_length = int((segment_length_ms / 1000) * sample_rate)
    step = int(segment_length * (1 - overlap))

    if waveform.shape[1] < segment_length:
        print("Audio too short, returning empty tensor.")
        return torch.empty(0, 1, segment_length)
    # Extract overlapping segments using `unfold`
    segments = waveform.unfold(1, segment_length, step).squeeze(0)

    return segments.unsqueeze(1)


def reconstruct_audio(segments, sample_rate, overlap=0.5):
    num_segments, _, segment_length = segments.shape
    step = int(segment_length * (1 - overlap))
    total_length = (num_segments - 1) * step + segment_length

    waveform = torch.zeros(1, total_length)
    waveform = waveform.to(segments.device)
    # Hann window for smooth overlap
    window = torch.hann_window(segment_length).unsqueeze(0)
    window = window.to(segments.device)
    for i in range(num_segments):
        start = i * step
        end = start + segment_length
        waveform[:, start:end] += segments[i] * window

    return waveform
