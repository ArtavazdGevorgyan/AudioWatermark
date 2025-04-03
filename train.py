import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from audiocraft.losses import TFLoudnessLoss

from loader import AudioSeal
from modules.dataloader import AudioDataLoader
# from builder import create_generator, create_detector
# from models import AudioSealWM, AudioSealDetector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = {
    "nbits": 32,
    "sample_rate": 16000,
    "batch_size": 32,
    "lr": 1e-4,
    "num_epochs": 100,
    "alpha": 0.1,
    "lambda_l1": 1.0,
    "lambda_mel": 1.0,
    "lambda_loudness": 1.0,
    "lambda_det": 1.0,
    "lambda_msg": 1.0,
}


# Loss Modules
class MelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_fn = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=1024, n_mels=80
        )
        self.loss = nn.L1Loss()

    def forward(self, original, watermarked):
        return self.loss(self.mel_fn(original), self.mel_fn(watermarked))


# class LoudnessLoss(nn.Module):
#     def __init__(self, frame_size=1024, hop_length=256):
#         super().__init__()
#         self.frame_size = frame_size
#         self.hop_length = hop_length
#         self.loss = nn.L1Loss()

#     def forward(self, original, watermarked):
#         def calc_rms(x):
#             return torch.sqrt(torch.mean(x**2, dim=-1))

#         orig_rms = calc_rms(original.unfold(-1, self.frame_size, self.hop_length))
#         wm_rms = calc_rms(watermarked.unfold(-1, self.frame_size, self.hop_length))
#         return self.loss(orig_rms, wm_rms)


# generator, discriminator, detector, dataloader,
generator = AudioSeal.load_generator("./cards/audioseal_wm_16bits.yaml", 16)
generator.to(device)
# discriminator.to(device)
generator.get_watermark(torch.ones((2, 1, 1000)), sample_rate=16000)

detector = AudioSeal.load_detector("./cards/audioseal_wm_16bits.yaml", 16)
detector.to(device)
# generator.train()
# discriminator.train()
# detector.train()


# Optimizers
opt = optim.Adam(generator.parameters(), lr=config["lr"])
# det_opt = optim.Adam(detector.parameters(), lr=config["lr"])

# Loss Functions
bce_loss = nn.BCELoss()
mel_criterion = MelLoss().to(device)
loudness_criterion = TFLoudnessLoss().to(device)
scaler = GradScaler()

# Data Loader
loader = AudioDataLoader("/path/to/audio", config["batch_size"])
train_loader = loader.get_dataloader()


# Training Loop
for epoch in range(config["num_epochs"]):
    for batch_idx, (reconstructed_audio, audio) in enumerate(train_loader):
        # real_audios = 
        # audios_to_watermark = 
    
        audio_to_watermark = audio.to(device)
        reconstructed_audio = reconstructed_audio.to(device)
        batch_size = audio_to_watermark.size(0)
        
        total_loss = 0
        watermark_flag = np.random.random() < 0.5
        if watermark_flag:
            # Generate random message
            msg = torch.randint(0, 2, (1, config["nbits"]), device=device).float()
            msg_repeated = msg.repeat(batch_size, 1)
            
            # --- Detector Training ---
            with torch.no_grad():
                generated_audio = generator(audio_to_watermark, config["sample_rate"], msg)
                # ^ does this wm_audio = audios_to_watermark + config["alpha"] * watermark
            
            reconstructed_gen_audio = 
            
            l1_loss = torch.mean(torch.abs(generated_audio - audio_to_watermark))
            mel_loss = mel_criterion(reconstructed_gen_audio, reconstructed_audio)
            loudness_loss = loudness_criterion(reconstructed_gen_audio, reconstructed_audio)
            total_loss += l1_loss + mel_loss + loudness_loss
            
            audio = generated_audio

        # # Detector forward
        # det_real, _ = detector(audios)
        is_watermarked_pred, msg_pred = detector(audio)

        # Detection loss
        if watermark_flag:
            gen_loss = torch.mean(torch.relu(1 - is_watermarked_pred))
            det_loss = torch.mean(torch.relu(1 + is_watermarked_pred))
            
            total_loss += gen_loss + det_loss
        else:
            det_loss = torch.mean(torch.relu(1 - is_watermarked_pred))
            total_loss += det_loss
        
        msg_loss = bce_loss(msg_pred, msg)
        total_loss += msg_loss

        # Update detector
        opt.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(opt)

        # --- Generator Training ---
        # Generate watermark
        # watermark = generator.get_watermark(audios_to_watermark, config["sample_rate"], msg)
        # wm_audio = audios_to_watermark + config["alpha"] * watermark

        # L1 Loss on watermark
        

        # Adversarial Losses
        # det_pred, msg_pred = detector(wm_audio)
        # det_loss_gen = bce_loss(
        #     det_pred[:, 1, :].mean(1), torch.ones(batch_size, device=device)
        # )
        # msg_loss_gen = bce_loss(msg_pred, msg)

        # # Total Generator Loss
        # total_gen_loss = (
        #     config["lambda_l1"] * l1_loss
        #     + config["lambda_mel"] * mel_loss
        #     + config["lambda_loudness"] * loudness_loss
        #     + config["lambda_det"] * det_loss_gen
        #     + config["lambda_msg"] * msg_loss_gen
        # )

        # # Update generator
        # gen_opt.zero_grad()
        # scaler.scale(total_gen_loss).backward()
        # scaler.step(gen_opt)
        # scaler.update()

        # # Logging
        # if batch_idx % 100 == 0:
        #     print(
        #         f"""Epoch {epoch+1}/{config["num_epochs"]} Batch {batch_idx}
        #         Det Loss: {total_det_loss.item():.4f}
        #         Gen Loss: {total_gen_loss.item():.4f}
        #         L1: {l1_loss.item():.4f} | Mel: {mel_loss.item():.4f}
        #         Loud: {loudness_loss.item():.4f} | DetG: {det_loss_gen.item():.4f}
        #         MsgG: {msg_loss_gen.item():.4f}"""
        #     )

# Save Models
torch.save(generator.state_dict(), "generator.pth")
torch.save(detector.state_dict(), "detector.pth")
