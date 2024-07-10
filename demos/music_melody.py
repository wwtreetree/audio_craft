import torchaudio
import argparse
from audiocraft.models.musicgen_shu import MusicGen
from audiocraft.data.audio import audio_write
from transformers import AutoTokenizer, AutoModelForCausalLM
from audiocraft.lib.prune_shu import prune_wanda, prune_magnitude, check_sparsity
import torch
from audiocraft.models import MusicGen
from audiocraft.models import MultiBandDiffusion
import math
import torchaudio
import torch
from audiocraft.utils.notebook import display_audio

USE_DIFFUSION_DECODER = False
if USE_DIFFUSION_DECODER:
    mbd = MultiBandDiffusion.get_mbd_musicgen()

model = MusicGen.get_pretrained('facebook/musicgen-melody')
model.set_generation_params(duration=8)

melody_waveform, sr = torchaudio.load("./assets/bach.mp3")
melody_waveform = melody_waveform.unsqueeze(0).repeat(2, 1, 1)
output = model.generate_with_chroma(
    descriptions = [
        '80s pop track with bassy drums and synth',
        '90s rock song with loud guitars and heavy drums',
    ],
    melody_wavs=melody_waveform,
    melody_sample_rate=sr,
    progress=True, return_tokens=True
)
display_audio(output[0], sample_rate=32000)
if USE_DIFFUSION_DECODER:
    out_diffusion = mbd.tokens_to_wav(output[1])
    display_audio(out_diffusion, sample_rate=32000)



parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='facebook/musicgen-melody', help='LLaMA model')
parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
parser.add_argument('--sparsity_ratio', type=float, default=0.1, help='Sparsity level')
parser.add_argument("--sparsity_type", type=str, default="unstructured", choices=["unstructured", "4:8", "2:4"])
parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", 
                    "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search"])
parser.add_argument("--cache_dir", default="llm_weights", type=str )
parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
parser.add_argument('--save', type=str, default=None, help='Path to save results.')
parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
parser.add_argument("--eval_zero_shot", action="store_true")
args = parser.parse_args()

device = torch.device("cuda:0")
tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
prune_n, prune_m = 0, 0

#prune_magnitude(args, model.lm, tokenizer, device, prune_n=0, prune_m=0)
#prune_wanda(args, model.lm, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

for idx, one_wav in enumerate(output):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    print("wav", one_wav.cpu().shape)
    one_wav = one_wav.squeeze()
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy= "loudness", loudness_compressor=True)
