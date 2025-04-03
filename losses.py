import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.functional import treble_biquad, highpass_biquad
from torch import autograd
import flashy
import math
import julius
from torchaudio.transforms import MelSpectrogram
import typing as tp


def _unfold(a: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    """Given input of size [*OT, T], output Tensor of size [*OT, F, K]
    with K the kernel size, by extracting frames with the given stride.
    This will pad the input so that `F = ceil(T / K)`.
    see https://github.com/pytorch/pytorch/issues/60466
    """
    shape = list(a.shape[:-1])
    length = int(a.shape[-1])
    n_frames = math.ceil(length / stride)
    tgt_length = (n_frames - 1) * stride + kernel_size
    a = F.pad(input=a, pad=[0, tgt_length - length])
    strides = [a.stride(dim) for dim in range(a.dim())]
    if strides[-1] != 1:
        raise ValueError("Data should be contiguous.")
    strides = strides[:-1] + [stride, 1]
    shape.append(n_frames)
    shape.append(kernel_size)
    return a.as_strided(shape, strides)


def pad_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    """Pad input tensor for 1D convolution to ensure output length matches
    input length divided by stride, rounded up.

    Args:
        x: Input tensor of shape (B, C, T).
        kernel_size: Size of the convolution kernel.
        stride: Stride of the convolution.
    """
    length = x.shape[-1]
    n_frames = (length + stride - 1) // stride
    desired_length = (n_frames - 1) * stride + kernel_size
    pad = max(0, desired_length - length)
    if pad > 0:
        x = F.pad(x, (0, pad))

    return x


def basic_loudness(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    This is a simpler loudness function that is more stable.
        Args:
            waveform(torch.Tensor): audio waveform of dimension `(..., channels, time)`
            sample_rate (int): sampling rate of the waveform
        Returns:
            loudness loss as a scalar
    """

    if waveform.size(-2) > 5:
        raise ValueError("Only up to 5 channels are supported.")
    eps = torch.finfo(torch.float32).eps
    gate_duration = 0.4
    overlap = 0.75
    gate_samples = int(round(gate_duration * sample_rate))
    step = int(round(gate_samples * (1 - overlap)))

    # Apply K-weighting
    waveform = treble_biquad(waveform, sample_rate, 4.0, 1500.0, 1 / math.sqrt(2))
    waveform = highpass_biquad(waveform, sample_rate, 38.0, 0.5)

    # Compute the energy for each block
    energy = torch.square(waveform).unfold(-1, gate_samples, step)
    energy = torch.mean(energy, dim=-1)

    # Compute channel-weighted summation
    g = torch.tensor(
        [1.0, 1.0, 1.0, 1.41, 1.41], dtype=waveform.dtype, device=waveform.device
    )
    g = g[: energy.size(-2)]

    energy_weighted = torch.sum(g.unsqueeze(-1) * energy, dim=-2)
    # loudness with epsilon for stability. Not as much precision in the very low loudness sections
    loudness = -0.691 + 10 * torch.log10(energy_weighted + eps)
    return loudness


class TFLoudnessRatio(nn.Module):
    """TF-loudness ratio loss implementation"""

    def __init__(
        self,
        sample_rate: int = 16000,
        segment: float = 0.5,
        overlap: float = 0.5,
        n_bands: int = 1,  ##########################es poxeluc ashxatec
        clip_min: float = -100,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.segment = segment
        self.overlap = overlap
        self.clip_min = clip_min
        self.temperature = temperature

        if n_bands == 0:
            self.filter = None
        else:
            self.n_bands = n_bands
            self.filter = julius.SplitBands(sample_rate=sample_rate, n_bands=n_bands)

    def forward(self, out_sig: torch.Tensor, ref_sig: torch.Tensor) -> torch.Tensor:
        B, C, T = ref_sig.shape
        assert ref_sig.shape == out_sig.shape
        assert C == 1
        assert self.filter is not None

        bands_ref = self.filter(ref_sig).view(B * self.n_bands, 1, -1)
        bands_out = self.filter(out_sig).view(B * self.n_bands, 1, -1)

        frame = int(self.segment * self.sample_rate)
        stride = int(frame * (1 - self.overlap))

        gt = (
            _unfold(bands_ref, frame, stride).squeeze(1).contiguous().view(-1, 1, frame)
        )
        est = (
            _unfold(bands_out, frame, stride).squeeze(1).contiguous().view(-1, 1, frame)
        )

        l_noise = basic_loudness(est - gt, sample_rate=self.sample_rate)
        l_ref = basic_loudness(gt, sample_rate=self.sample_rate)
        l_ratio = (l_noise - l_ref).view(-1, B)
        loss = F.softmax(l_ratio / self.temperature, dim=0) * l_ratio
        return loss.mean()


class MelSpectrogramWrapper(nn.Module):
    """
    Wrapper around MelSpectrogram torchaudio transform providing proper padding
        and additional post-processing including log scaling.

        Args:
            n_mels (int): Number of mel bins.
            n_fft (int): Number of fft.
            hop_length (int): Hop size.
            win_length (int): Window length.
            n_mels (int): Number of mel bins.
            sample_rate (int): Sample rate.
            f_min (float or None): Minimum frequency.
            f_max (float or None): Maximum frequency.
            log (bool): Whether to scale with log.
            normalized (bool): Whether to normalize the melspectrogram.
            floor_level (float): Floor level based on human perception (default=1e-5).
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: tp.Optional[int] = None,
        n_mels: int = 80,
        sample_rate: float = 22050,
        f_min: float = 0.0,
        f_max: tp.Optional[float] = None,
        log: bool = True,
        normalized: bool = False,
        floor_level: float = 1e-5,
    ):
        super().__init__()
        self.n_fft = n_fft
        hop_length = int(hop_length)
        self.hop_length = hop_length
        self.mel_transform = MelSpectrogram(
            n_mels=n_mels,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            f_min=f_min,
            f_max=f_max,
            normalized=normalized,
            window_fn=torch.hann_window,
            center=False,
        )
        self.floor_level = floor_level
        self.log = log

    def forward(self, x):
        p = int((self.n_fft - self.hop_length) // 2)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = F.pad(x, (p, p), "reflect")
        # Make sure that all the frames are full.
        # The combination of `pad_for_conv1d` and the above padding
        # will make the output of size ceil(T / hop).
        x = pad_for_conv1d(x, self.n_fft, self.hop_length)
        self.mel_transform.to(x.device)
        mel_spec = self.mel_transform(x)
        B, C, freqs, frame = mel_spec.shape
        if self.log:
            mel_spec = torch.log10(self.floor_level + mel_spec)
        return mel_spec.reshape(B, C * freqs, frame)


class MelSpectrogramL1Loss(torch.nn.Module):
    """
    L1 Loss on MelSpectrogram.

        Args:
            sample_rate (int): Sample rate.
            n_fft (int): Number of fft.
            hop_length (int): Hop size.
            win_length (int): Window length.
            n_mels (int): Number of mel bins.
            f_min (float or None): Minimum frequency.
            f_max (float or None): Maximum frequency.
            log (bool): Whether to scale with log.
            normalized (bool): Whether to normalize the melspectrogram.
            floor_level (float): Floor level value based on human perception (default=1e-5).
    """

    def __init__(
        self,
        sample_rate: int,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: tp.Optional[float] = None,
        log: bool = True,
        normalized: bool = False,
        floor_level: float = 1e-5,
    ):
        super().__init__()
        self.l1 = torch.nn.L1Loss()
        self.melspec = MelSpectrogramWrapper(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
            log=log,
            normalized=normalized,
            floor_level=floor_level,
        )

    def forward(self, x, y):
        self.melspec.to(x.device)
        s_x = self.melspec(x)
        s_y = self.melspec(y)
        return self.l1(s_x, s_y)


class Balancer:
    def __init__(
        self,
        weights: tp.Dict[str, float],
        balance_grads: bool = True,
        total_norm: float = 1.0,
        ema_decay: float = 0.999,
        per_batch_item: bool = True,
        epsilon: float = 1e-12,
        monitor: bool = False,
    ):
        self.weights = weights
        self.per_batch_item = per_batch_item
        self.total_norm = total_norm or 1.0
        self.averager = flashy.averager(ema_decay or 1.0)
        self.epsilon = epsilon
        self.monitor = monitor
        self.balance_grads = balance_grads
        self._metrics: tp.Dict[str, tp.Any] = {}

    @property
    def metrics(self):
        return self._metrics

    def backward(
        self, losses: tp.Dict[str, torch.Tensor], input: torch.Tensor
    ) -> torch.Tensor:

        norms = {}
        grads = {}
        for name, loss in losses.items():
            # Compute partial derivative of the less with respect to the input.
            (grad,) = autograd.grad(loss, [input], retain_graph=True)
            if self.per_batch_item:
                # We do not average the gradient over the batch dimension.
                dims = tuple(range(1, grad.dim()))
                norm = grad.norm(dim=dims, p=2).mean()
            else:
                norm = grad.norm(p=2)
            norms[name] = norm
            grads[name] = grad

        count = 1
        if self.per_batch_item:
            count = len(grad)
        # Average norms across workers. Theoretically we should average the
        # squared norm, then take the sqrt, but it worked fine like that.
        avg_norms = flashy.distrib.average_metrics(self.averager(norms), count)
        # We approximate the total norm of the gradient as the sums of the norms.
        # Obviously this can be very incorrect if all gradients are aligned, but it works fine.
        total = sum(avg_norms.values())

        self._metrics = {}
        if self.monitor:
            # Store the ratio of the total gradient represented by each loss.
            for k, v in avg_norms.items():
                self._metrics[f"ratio_{k}"] = v / total

        total_weights = sum([self.weights[k] for k in avg_norms])
        assert total_weights > 0.0
        desired_ratios = {k: w / total_weights for k, w in self.weights.items()}

        out_grad = torch.zeros_like(input)
        effective_loss = torch.tensor(0.0, device=input.device, dtype=input.dtype)
        for name, avg_norm in avg_norms.items():
            if self.balance_grads:
                # g_balanced = g / avg(||g||) * total_norm * desired_ratio
                scale = (
                    desired_ratios[name] * self.total_norm / (self.epsilon + avg_norm)
                )
            else:
                # We just do regular weighted sum of the gradients.
                scale = self.weights[name]
            out_grad.add_(grads[name], alpha=scale)
            effective_loss += scale * losses[name].detach()
        # Send the computed partial derivative with respect to the output of the model to the model.
        input.backward(out_grad)
        return effective_loss
