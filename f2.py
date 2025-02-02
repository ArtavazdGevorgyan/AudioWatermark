import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def compute_spectrogram_loss(x, x_hat, alpha=1.0):
    """
    Compute frequency-domain loss as a combination of L1 and L2 losses.
    """
    loss = 0.0
    for scale in range(5, 12):  # Corresponds to 64-bin mel-spectrogram scales
        mel_x = compute_mel_spectrogram(x, scale)
        mel_x_hat = compute_mel_spectrogram(x_hat, scale)
        loss += torch.mean(torch.abs(mel_x - mel_x_hat)) + alpha * torch.mean(
            (mel_x - mel_x_hat) ** 2
        )
    return loss / (12 - 5)


def fit(
    generator,
    discriminator,
    dataloader,
    num_epochs,
    device,
    g_optimizer,
    d_optimizer,
    lambda_t=1,
    lambda_f=1,
    lambda_g=1,
    lambda_feat=1,
    lambda_w=1,
    checkpoint_path=None,
):
    """
    Train the GAN model with the updated losses.
    """
    generator.to(device)
    discriminator.to(device)
    generator.train()
    discriminator.train()

    criterion = nn.MSELoss()  # For reconstruction loss
    commitment_loss_fn = nn.MSELoss()  # For latent commitment loss

    for epoch in range(num_epochs):
        g_loss_epoch, d_loss_epoch = 0, 0

        for real_audio, labels in tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"
        ):
            real_audio, labels = real_audio.to(device), labels.to(device)

            # Train discriminator
            d_optimizer.zero_grad()
            fake_audio = generator(labels)
            real_pred = discriminator(real_audio)
            fake_pred = discriminator(fake_audio.detach())
            d_loss = torch.mean(torch.clamp(1 - real_pred, min=0)) + torch.mean(
                torch.clamp(1 + fake_pred, min=0)
            )
            d_loss.backward()
            d_optimizer.step()

            # Train generator
            g_optimizer.zero_grad()
            fake_pred = discriminator(fake_audio)

            # Compute losses
            time_loss = criterion(
                real_audio, fake_audio
            )  # Time domain reconstruction loss
            freq_loss = compute_spectrogram_loss(
                real_audio, fake_audio
            )  # Frequency domain loss
            adv_loss = torch.mean(
                torch.clamp(1 - fake_pred, min=0)
            )  # Generator adversarial loss
            feat_loss = compute_feature_matching_loss(
                real_audio, fake_audio, discriminator
            )  # Feature matching loss
            commit_loss = commitment_loss_fn(
                generator.latent, generator.quantized_latent
            )  # Latent commitment loss

            # Combine losses
            g_loss = (
                lambda_t * time_loss
                + lambda_f * freq_loss
                + lambda_g * adv_loss
                + lambda_feat * feat_loss
                + lambda_w * commit_loss
            )
            g_loss.backward()
            g_optimizer.step()

            # Accumulate losses
            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss.item()

        # Log epoch losses
        print(
            f"Epoch {epoch + 1}/{num_epochs} - Generator Loss: {g_loss_epoch:.4f}, Discriminator Loss: {d_loss_epoch:.4f}"
        )

        # Save checkpoints
        if checkpoint_path:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "generator_state_dict": generator.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "g_optimizer_state_dict": g_optimizer.state_dict(),
                    "d_optimizer_state_dict": d_optimizer.state_dict(),
                },
                f"{checkpoint_path}/checkpoint_epoch_{epoch + 1}.pth",
            )
