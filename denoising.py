import torch
import torchaudio
import torch.nn.functional as F
from training_model import config, UNetGenerator

input_audio_path = "C:/Users/zhang/Documents/codes/AudioDenoising/input_audio.wav"
model_path = "C:/Users/zhang/Documents/codes/AudioDenoising/model.pt"
output_audio_path = "C:/Users/zhang/Documents/codes/AudioDenoising/output_audio.wav"

# Load the audio file
signal_noisy, sr = torchaudio.load(input_audio_path)

# Preprocess the audio
if sr != config.target_sample_rate:
    resampler = torchaudio.transforms.Resample(sr, config.target_sample_rate)
    signal_noisy = resampler(signal_noisy)

if signal_noisy.shape[0] > 1:
    signal_noisy = torch.mean(signal_noisy, axis=0, keepdim=True)

if signal_noisy.shape[1] < config.target_sample_rate * config.duration:
    num_missing_samples = (config.target_sample_rate * config.duration) - signal_noisy.shape[1]
    signal_noisy = F.pad(signal_noisy, (0, num_missing_samples))

# Convert to Mel spectrogram
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=config.target_sample_rate,
    n_fft=config.n_fft,
    hop_length=config.hop_length,
    n_mels=config.n_mels,
)
mel_noisy = mel_spectrogram(signal_noisy)

# Load the trained model
model = UNetGenerator()
model.load_state_dict(torch.load(model_path))
model.eval()

# Denoise the audio
with torch.no_grad():
    mel_clean_pred = model(mel_noisy.unsqueeze(0))

# Convert back to waveform
griffin_lim = torchaudio.transforms.GriffinLim(
    n_fft=config.n_fft, hop_length=config.hop_length, n_iter=32, power=1.0
)
signal_clean_pred = griffin_lim(mel_clean_pred.squeeze(0))

# Save the denoised audio file
torchaudio.save(output_audio_path, signal_clean_pred, config.target_sample_rate)