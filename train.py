import torch.nn.functional as F
import torch
from loader import AudioSeal
from torchsummary import summary
import numpy as np

###
### TFLoudnessLoss
###
# def compute_loudness_loss(watermark, original, window_size, overlap, bands):
#     """
#     Compute time-frequency loudness loss based on auditory masking.
#     """
#     # Divide signals into non-overlapping frequency bands
#     watermarked_bands = divide_into_bands(watermark, bands)
#     original_bands = divide_into_bands(original, bands)

#     loss = 0.0
#     for b in range(bands):
#         wm_segments = segment_signal(watermarked_bands[b], window_size, overlap)
#         orig_segments = segment_signal(original_bands[b], window_size, overlap)

#         for wm_seg, orig_seg in zip(wm_segments, orig_segments):
#             lw_b = loudness(wm_seg) - loudness(orig_seg)
#             loss += F.softmax(lw_b, dim=0) * lw_b
#     return loss


def compute_localization_loss(detection_output, ground_truth):
    """
    Compute sample-level binary cross-entropy (BCE) localization loss.
    """
    return F.binary_cross_entropy(detection_output, ground_truth)


def fit_extended(
    num_epochs,
    device,
    # g_optimizer,
    # d_optimizer,
    lambda_t=1,
    lambda_f=1,
    lambda_g=1,
    lambda_feat=1,
    lambda_w=1,
    lambda_loud=1,
    lambda_loc=1,
    checkpoint_path=None,
):
    """
    Train the GAN model with additional perceptual, loudness, and localization losses.
    """

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

    criterion = nn.MSELoss()  # For reconstruction loss
    commitment_loss_fn = nn.MSELoss()  # For latent commitment loss

    for epoch in range(num_epochs):
        g_loss_epoch, d_loss_epoch = 0, 0

        for real_audio, labels, ground_truth in tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"
        ):
            real_audio, labels, ground_truth = (
                real_audio.to(device),
                labels.to(device),
                ground_truth.to(device),
            )

            # Train discriminator
            d_optimizer.zero_grad()
            fake_audio = generator(labels, real_audio)
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

            # Compute perceptual losses
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

            # Compute loudness and localization losses
            loudness_loss = compute_loudness_loss(
                fake_audio, real_audio, window_size=1024, overlap=256, bands=8
            )
            detection_output = detector(fake_audio)
            localization_loss = compute_localization_loss(
                detection_output, ground_truth
            )

            # Combine losses
            g_loss = (
                lambda_t * time_loss
                + lambda_f * freq_loss
                + lambda_g * adv_loss
                + lambda_feat * feat_loss
                + lambda_w * commit_loss
                + lambda_loud * loudness_loss
                + lambda_loc * localization_loss
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
                    "detector_state_dict": detector.state_dict(),
                    "g_optimizer_state_dict": g_optimizer.state_dict(),
                    "d_optimizer_state_dict": d_optimizer.state_dict(),
                },
                f"{checkpoint_path}/checkpoint_epoch_{epoch + 1}.pth",
            )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


fit_extended(num_epochs=5, device=device)
