# code in development (a language model (LM) that auto-regressively generates audio tokens (or codes) conditional on the text encoder
# hidden-state representation)

import torch
import torch.nn as nn
import librosa

from argparse import ArgumentParser

class ResidualVectorQuantizer(nn.Module):
    def __init__(self, num_codebooks, codebooks_size, features_dim, device):
        super(ResidualVectorQuantizer, self).__init__()
        self.num_codebooks = num_codebooks
        self.codebooks = nn.Parameter(torch.rand(num_codebooks, codebooks_size, features_dim))
    def quantize(self, x):
        residual = x.clone()
        quantized_output = torch.zeros_like(x)
        quantized_indices = []
        for i in range(self.num_codebooks):
            distances = torch.cdist(residual, self.codebooks[i])
            indices = distances.argmin(dim=1)
            quantized_vectors = self.codebooks[i][indices]
            quantized_output += quantized_vectors
            residual -= quantized_vectors
            quantized_indices.append(indices)
        return quantized_output, torch.stack(quantized_indices, dim=1)
    def dequantize(self, quantize_indices):
        reconstructed = torch.zeros((quantize_indices.shape[0], self.codebooks.shape[2]))
        for i in range(self.num_codebooks):
            reconstructed += self.codebooks[i][quantize_indices[:, i]]
        return reconstructed

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wave, sample_rate = librosa.load(args.input_music_dir, args.sample_rate)
    music_spectogram = librosa.feature.melspectrogram(
        y=wave,
        sr=sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels
    )
    music_spectogram_normalize = librosa.amplitude_to_db(music_spectogram)
    music_spectogram_normalize_transpose = music_spectogram_normalize.T
    music_tensor_data = torch.tensor(music_spectogram_normalize_transpose).to(device)
    rvq_model = ResidualVectorQuantizer(args.num_codebooks, args.codebooks_size, args.n_mels).to(device)
    rvq_model.load_state_dict(torch.load(args.best_rvq_model_dir))
    rvq_model.eval()
    with torch.no_grad():
        music_quantized, quantized_indices = rvq_model.quantize(music_tensor_data)
    print(f"Quantized Music Audio Data: \n{music_quantized}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_music_dir", type=str, default="./music.wav")
    parser.add_argument("--output_music_dir", type=str, default="./music_quantized.wav")
    parser.add_argument("--best_rvq_model_dir", type=str, default="./best_rvq_model.pth")
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--num_mels", type=int, default=64)
    parser.add_argument("--num_codebooks", type=int, default=4)
    parser.add_argument("--codebooks_size", type=int, default=512)
    args = parser.parse_args()
    print(args)
    main(args)