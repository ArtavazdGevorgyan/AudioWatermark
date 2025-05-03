import os
import torch
import glob
import torch.nn as nn
import torch.optim as optim
from losses import TFLoudnessRatio, MelSpectrogramL1Loss, Balancer
from loader import AudioSeal
from modules.dataloader import create_dataloader
from split_reconstruct_audio import reconstruct_audio
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


config = {
    "nbits": 32,
    "sample_rate": 16000,
    "batch_size": 32,
    "lr": 1e-3,
    "num_epochs": 100,
    "alpha": 0.1,
}

weights = {
    "l1_loss": 0.1,
    "mel_loss": 2.0,
    "loudness_loss": 10.0,
    "gen_loss": 4.0,
    "det_loss": 4.0,
    "msg_loss": 10.0,
}

device = torch.device(
    "mps"  # Or "mps" if torch.backends.mps.is_available() else "cuda"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)


def save_checkpoint(
    epoch, generator, detector, gen_opt, det_opt, loss, checkpoint_dir, is_best=False
):
    checkpoint = {
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "detector_state_dict": detector.state_dict(),
        "gen_opt_state_dict": gen_opt.state_dict(),
        "det_opt_state_dict": det_opt.state_dict(),
        "loss": loss,
    }
    os.makedirs(checkpoint_dir, exist_ok=True)

    if is_best:
        torch.save(checkpoint, os.path.join(checkpoint_dir, "best_checkpoint.pt"))
    else:
        torch.save(
            checkpoint, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        )
    print(f"Checkpoint saved at epoch {epoch}")


def load_checkpoint(checkpoint_path, generator, detector, gen_opt, det_opt):
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    detector.load_state_dict(checkpoint["detector_state_dict"])
    gen_opt.load_state_dict(checkpoint["gen_opt_state_dict"])
    det_opt.load_state_dict(checkpoint["det_opt_state_dict"])
    start_epoch = checkpoint["epoch"] + 1  # Start from next epoch
    best_loss = checkpoint.get("loss", float("inf"))
    print(f"Resuming training from epoch {start_epoch}")
    return start_epoch, best_loss


def find_latest_checkpoint(checkpoint_dir):
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt"))
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    return latest_checkpoint


# Model loading with error handling
try:
    generator = AudioSeal.load_generator("./cards/audioseal_wm_16bits.yaml", 16)
    generator.to(device)
    detector = AudioSeal.load_detector("./cards/audioseal_detector_16bits.yaml", 16)
    detector.to(device)
except Exception as e:
    raise RuntimeError(f"Failed to load models: {str(e)}")

# Optimizers
gen_opt = optim.Adam(generator.parameters(), lr=config["lr"])
det_opt = optim.Adam(detector.parameters(), lr=config["lr"])

# Loss functions
mel_criterion = MelSpectrogramL1Loss(sample_rate=config["sample_rate"]).to(device)
loudness_criterion = TFLoudnessRatio().to(device)
bce_loss = nn.BCELoss()
cat_cross_loss = nn.CrossEntropyLoss()

checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

global moving_accuracy
moving_accuracy = 0.0

# Try to load latest checkpoint
start_epoch = 0
best_val_loss = float("inf")
latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    start_epoch, best_val_loss = load_checkpoint(
        latest_checkpoint, generator, detector, gen_opt, det_opt
    )

writer = SummaryWriter(log_dir="runs/audioseal_experiment")
gen_balancer = Balancer(weights=weights)
raw_balancer = Balancer(weights={"det_loss": 1.0})


def batch_step(
    watermark_flag,
    raw_audio,
    segments,
    msg,
    reconstructed_audio,
    labels,
    step,
    sample_rate,
    generator,
    detector,
    raw_balancer,
    gen_balancer,
    train=True,
):
    loss_functions = {}

    # Ensure tensors are on the correct device
    raw_audio = raw_audio.to(device).requires_grad_(train)
    segments = (
        segments.to(device).requires_grad_(train) if segments is not None else None
    )
    msg = msg.to(device).requires_grad_(train) if msg is not None else None
    labels = labels.to(device)
    reconstructed_audio = reconstructed_audio.to(device).requires_grad_(train)

    if watermark_flag and segments is not None:
        with torch.set_grad_enabled(train):
            generated_audios = generator(segments, sample_rate, msg)
            reconstructed_audio = reconstruct_audio(generated_audios, sample_rate)
            reconstructed_audio = reconstructed_audio.requires_grad_(train)

            # Ensure matching dimensions
            min_length = min(reconstructed_audio.shape[1], raw_audio.shape[1])
            raw_audio = raw_audio[:, :min_length]
            reconstructed_audio = reconstructed_audio[:, :min_length]

            # Calculate losses
            l1_loss = weights["l1_loss"] * torch.mean(
                torch.abs(reconstructed_audio - raw_audio)
            )
            mel_loss = weights["mel_loss"] * mel_criterion(
                reconstructed_audio, raw_audio
            )
            loudness_loss = weights["loudness_loss"] * loudness_criterion(
                reconstructed_audio.unsqueeze(0),
                raw_audio.unsqueeze(0),
            )

            loss_functions.update(
                {
                    "l1_loss": l1_loss,
                    "mel_loss": mel_loss,
                    "loudness_loss": loudness_loss,
                }
            )

    # Detection part
    with torch.set_grad_enabled(train):
        is_watermarked_pred, msg_pred = detector(
            reconstructed_audio.unsqueeze(0), sample_rate=sample_rate
        )

        labels_acc = labels.unsqueeze(1).repeat(1, is_watermarked_pred.shape[2])
        tmp = labels_acc * torch.round(is_watermarked_pred.squeeze(0))
        print(f"Detector Accuracy: {torch.mean(tmp)*2:.4f}")

        # Properly handle predictions
        is_watermarked_pred = is_watermarked_pred.to(device)
        det_loss = weights["det_loss"] * cat_cross_loss(
            torch.mean(is_watermarked_pred, axis=2), labels.float().unsqueeze(0)
        )
        loss_functions["det_loss"] = det_loss

        if watermark_flag and msg is not None:
            gen_loss = weights["gen_loss"] * cat_cross_loss(
                torch.mean(is_watermarked_pred, axis=2), 1 - labels.float().unsqueeze(0)
            )
            msg_loss = weights["msg_loss"] * cat_cross_loss(
                msg_pred, msg[0].unsqueeze(0)
            )
            loss_functions.update({"gen_loss": gen_loss, "msg_loss": msg_loss})

            # Generator update
            total_loss = gen_balancer.backward(
                loss_functions, raw_audio, reconstructed_audio, segments, msg, train
            )

            accuracy = ((msg_pred > 0.5).float() == msg).float().mean().item()

            global moving_accuracy

            if step != 1:
                moving_accuracy = moving_accuracy * (step - 1) / step
            moving_accuracy += accuracy / step

            print(f"Message Accuracy: {accuracy:.4f}")
            print(f"Message Accuracy moving: {moving_accuracy:.4f}\n")
            writer.add_scalar(f"Accuracy/{'train' if train else 'val'}", accuracy, step)

            if train:
                gen_opt.zero_grad()
                # total_loss.backward()
                gen_opt.step()
        else:
            # Detector update
            total_loss = raw_balancer.backward(
                loss_functions, raw_audio, reconstructed_audio, segments, msg, train
            )
            if train:
                gen_opt.zero_grad()
                # total_loss.backward()
                gen_opt.step()

    # Logging
    writer.add_scalar(f"Loss/{'train' if train else 'val'}", total_loss.item(), step)
    return total_loss.item()


# Data loading with error handling
try:
    train_loader, val_loader = create_dataloader(
        "/Users/artavazdgevorgyan/Desktop/untitled folder"
    )
except Exception as e:
    raise RuntimeError(f"Failed to load data: {str(e)}")

# Training setup
val_interval = 10000
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

writer = SummaryWriter(log_dir="runs/audioseal_experiment")
gen_balancer = Balancer(weights=weights)
raw_balancer = Balancer(weights={"det_loss": 10.0})

# Training loop
step = 0
best_val_loss = float("inf")

for epoch in range(start_epoch, config["num_epochs"]):
    generator.train()
    detector.train()
    total_epoch_loss = 0

    for idx in tqdm(range(len(train_loader)), desc=f"Epoch {epoch+1}"):
        try:
            raw_audio, segments, msg, watermark_flag = train_loader[idx]
            labels = torch.tensor([1 - watermark_flag, watermark_flag], device=device)
            step += 1

            loss = batch_step(
                watermark_flag,
                raw_audio.unsqueeze(0),
                segments,
                msg,
                raw_audio.unsqueeze(0),  # Initial reconstruction is just the input
                labels,
                step,
                config["sample_rate"],
                generator,
                detector,
                raw_balancer,
                gen_balancer,
                train=True,
            )
            total_epoch_loss += loss

            # Validation
            if step % val_interval == 0:
                generator.eval()
                detector.eval()
                val_loss = 0
                with torch.no_grad():
                    for val_idx in range(
                        min(10, len(val_loader))
                    ):  # Validate on 10 samples
                        raw_audio, segments, msg, watermark_flag = val_loader[val_idx]
                        labels = torch.tensor(
                            [1 - watermark_flag, watermark_flag], device=device
                        )

                        loss = batch_step(
                            watermark_flag,
                            raw_audio.unsqueeze(0),
                            segments,
                            msg,
                            raw_audio.unsqueeze(0),
                            labels,
                            step,
                            config["sample_rate"],
                            generator,
                            detector,
                            raw_balancer,
                            gen_balancer,
                            train=False,
                        )
                        val_loss += loss

                avg_val_loss = val_loss / min(10, len(val_loader))

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    save_checkpoint(
                        epoch,
                        generator,
                        detector,
                        gen_opt,
                        det_opt,
                        avg_val_loss,
                        checkpoint_dir,
                        is_best=True,
                    )

                print(f"\nValidation Loss: {avg_val_loss:.4f}")

        except Exception as e:
            print(f"\nError in batch {idx}: {str(e)}")
            continue

    # Save periodic checkpoint
    torch.save(
        {
            "epoch": epoch,
            "generator_state_dict": generator.state_dict(),
            "detector_state_dict": detector.state_dict(),
            "gen_opt_state_dict": gen_opt.state_dict(),
            "det_opt_state_dict": det_opt.state_dict(),
            "loss": total_epoch_loss / len(train_loader),
        },
        os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"),
    )

    print(f"Epoch {epoch+1} Average Loss: {total_epoch_loss / len(train_loader):.4f}")

writer.close()
