import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from losses import TFLoudnessRatio, MelSpectrogramL1Loss, Balancer
from loader import AudioSeal
from modules.dataloader import AudioWatermarkDataset, create_dataloader
from split_reconstruct_audio import split_audio, reconstruct_audio
from tqdm import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
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


generator = AudioSeal.load_generator("./cards/audioseal_wm_16bits.yaml", 16)
generator.to(device)
detector = AudioSeal.load_detector("./cards/audioseal_wm_16bits.yaml", 16)
detector.to(device)
gen_opt = optim.Adam(generator.parameters(), lr=config["lr"])
det_opt = optim.Adam(detector.parameters(), lr=config["lr"])
mel_criterion = MelSpectrogramL1Loss(sample_rate=16000).to(device)
loudness_criterion = TFLoudnessRatio().to(device)
bce_loss = nn.BCELoss()
# scaler = GradScaler()

# train_loader = AudioWatermarkDataset(
#     "/Users/artavazdgevorgyan/Downloads/audioseal_dataset/WAV_format/train_wav",
#     config["batch_size"],
# )
train_loader = create_dataloader(
    "/Users/artavazdgevorgyan/Downloads/audioseal_dataset/WAV_format/train_wav"
)

weights = {
    "l1_loss": 1.0,
    "mel_loss": 1.0,
    "loudness_loss": 1.0,
    "gen_loss": 1.0,
    "det_loss": 1.0,
    "msg_loss": 1.0,
}
gen_balancer = Balancer(weights=weights)
raw_balancer = Balancer(weights={"det_loss": 1.0})

for epoch in range(config["num_epochs"]):
    total_epoch_loss = 0
    for idx in tqdm(range(len(train_loader))):
        print(1)
        raw_audio, segments, msg, watermark_flag = train_loader.__getitem__(idx)
        print(2)
        labels = torch.full((len(segments),), 1 - watermark_flag)
        print(3)
        reconstructed_audio = raw_audio.to(device).requires_grad_()
        print(4)
        raw_audio = raw_audio.unsqueeze(0).to(device).requires_grad_()
        print(5)
        msg = msg.to(device).requires_grad_()
        # batch_size = len(segments)
        # audio_to_watermark = audio.to(device)
        # reconstructed_audio = reconstructed_audio.to(device)
        # batch_size = audio_to_watermark.size(0)

        gradients = []
        loss_functions = {}

        if True:
            print("Watermarking")
            segments = segments.to(device)

            with torch.no_grad():
                generated_audios = generator(segments, config["sample_rate"], msg)

            print("something")
            reconstructed_audio = reconstruct_audio(generated_audios, 16000)
            reconstructed_audio = reconstructed_audio.requires_grad_()
            # reconstructed_audio = reconstructed_audio.to(device)
            print(reconstructed_audio.shape)
            print(raw_audio.shape)

            l1_loss = torch.mean(torch.abs(reconstructed_audio - raw_audio))
            mel_loss = mel_criterion(reconstructed_audio, raw_audio)
            loudness_loss = loudness_criterion(
                reconstructed_audio.unsqueeze(0),
                raw_audio.unsqueeze(0),  ######### ES Harca
            )

            loss_functions["l1_loss"] = l1_loss
            loss_functions["mel_loss"] = mel_loss
            loss_functions["loudness_loss"] = loudness_loss

        is_watermarked_pred, msg_pred = detector(
            reconstructed_audio.unsqueeze(0)
        )  # es el harc toxenq
        # det_loss = bce_loss(is_watermarked_pred, labels)##################es uncomment
        # gen_loss = torch.mean(torch.relu(1 - is_watermarked_pred))
        # det_loss = torch.mean(torch.relu(1 + is_watermarked_pred))
        # loss_functions["det_loss"] = det_loss##################es uncomment

        if watermark_flag:
            # gen_loss = bce_loss(is_watermarked_pred, 1 - labels)##################es uncomment
            msg_pred = msg_pred.to(device).requires_grad_()

            msg_loss = bce_loss(msg_pred.squeeze(0), msg[0])
            # loss_functions["gen_loss"] = gen_loss##################es uncomment
            loss_functions["msg_loss"] = msg_loss

        if watermark_flag:
            total_loss = gen_balancer.backward(loss_functions, reconstructed_audio)
            gen_opt.zero_grad()
            gen_opt.step()
        else:
            total_loss = raw_balancer.backward(loss_functions, reconstructed_audio)

        det_opt.zero_grad()
        det_opt.step()

        del raw_audio, segments, msg, reconstructed_audio, msg_pred, is_watermarked_pred
        print(idx)
        # if idx % 100 == 0:
        #     print(
        #         f"Epoch [{epoch+1}/{config['num_epochs']}], Batch [{idx}], Loss: {batch_loss:.4f}"
        #     )

    print(f"Epoch [{epoch+1}/{config['num_epochs']}], Total Loss: {total_loss:.4f}")
