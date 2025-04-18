import torch
import torch.nn as nn
import torch.optim as optim
from losses import TFLoudnessRatio, MelSpectrogramL1Loss, Balancer
from loader import AudioSeal
from modules.dataloader import create_dataloader
from split_reconstruct_audio import reconstruct_audio
from tqdm import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
config = {
    "nbits": 32,
    "sample_rate": 16_000,
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
mel_criterion = MelSpectrogramL1Loss(sample_rate=config["sample_rate"]).to(device)
loudness_criterion = TFLoudnessRatio().to(device)
bce_loss = nn.BCELoss()


train_loader, val_loader = create_dataloader(
    "/Users/artavazdgevorgyan/Downloads/audioseal_dataset/train_wav"
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
        raw_audio, segments, msg, watermark_flag = train_loader.__getitem__(idx)
        labels = (
            torch.tensor([1 - watermark_flag, watermark_flag])
            .to(device)
            .requires_grad_()
        )
        raw_audio = raw_audio.unsqueeze(0).to(device).requires_grad_()
        reconstructed_audio = raw_audio.to(device).requires_grad_()
        msg = msg.to(device).requires_grad_()

        gradients = []
        loss_functions = {}

        if watermark_flag:
            print("Watermarking")
            segments = segments.to(device)

            with torch.no_grad():
                generated_audios = generator(segments, config["sample_rate"], msg)

            print("something")
            reconstructed_audio = reconstruct_audio(
                generated_audios, config["sample_rate"]
            )
            reconstructed_audio = reconstructed_audio.requires_grad_()
            raw_audio = raw_audio[:, : reconstructed_audio.shape[1]].requires_grad_()
            print(reconstructed_audio.shape)
            print(raw_audio.shape)

            l1_loss = torch.mean(torch.abs(reconstructed_audio - raw_audio))
            mel_loss = mel_criterion(reconstructed_audio, raw_audio)
            loudness_loss = loudness_criterion(
                reconstructed_audio.unsqueeze(0),
                raw_audio.unsqueeze(0),
            )

            loss_functions["l1_loss"] = l1_loss
            loss_functions["mel_loss"] = mel_loss
            loss_functions["loudness_loss"] = loudness_loss

        print(reconstructed_audio.shape)
        print(raw_audio.shape)

        is_watermarked_pred, msg_pred = detector(
            reconstructed_audio.unsqueeze(0), sample_rate=config["sample_rate"]
        )
        is_watermarked_pred = is_watermarked_pred.to(device).requires_grad_()
        det_loss = bce_loss(torch.mean(is_watermarked_pred.squeeze(0), axis=1), labels)
        loss_functions["det_loss"] = det_loss

        if watermark_flag:
            gen_loss = bce_loss(
                torch.mean(is_watermarked_pred.squeeze(0), axis=1), 1 - labels
            )
            msg_pred = msg_pred.to(device).requires_grad_()

            msg_loss = bce_loss(msg_pred.squeeze(0), msg[0])
            loss_functions["gen_loss"] = gen_loss
            loss_functions["msg_loss"] = msg_loss

        if watermark_flag:
            total_loss = gen_balancer.backward(loss_functions, reconstructed_audio)
            gen_opt.zero_grad()
            gen_opt.step()
        else:
            total_loss = raw_balancer.backward(loss_functions, reconstructed_audio)

        det_opt.zero_grad()
        det_opt.step()

        torch.mps.empty_cache()
        print(idx)

    print(f"Epoch [{epoch+1}/{config['num_epochs']}], Total Loss: {total_loss:.4f}")
