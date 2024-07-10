# from audiocraft.models import MusicGen
# from audiocraft.models import MultiBandDiffusion

# USE_DIFFUSION_DECODER = False
# # Using small model, better results would be obtained with `medium` or `large`.
# model = MusicGen.get_pretrained('facebook/musicgen-small')
# if USE_DIFFUSION_DECODER:
#     mbd = MultiBandDiffusion.get_mbd_musicgen()


# model.set_generation_params(
#     use_sampling=True,
#     top_k=250,
#     duration=30
# )

# import math
# import torchaudio
# import torch
# from audiocraft.utils.notebook import display_audio

# def get_bip_bip(bip_duration=0.125, frequency=440,
#                 duration=0.5, sample_rate=32000, device="cuda"):
#     """Generates a series of bip bip at the given frequency."""
#     t = torch.arange(
#         int(duration * sample_rate), device="cuda", dtype=torch.float) / sample_rate
#     wav = torch.cos(2 * math.pi * 440 * t)[None]
#     tp = (t % (2 * bip_duration)) / (2 * bip_duration)
#     envelope = (tp >= 0.5).float()
#     return wav * envelope


# # Here we use a synthetic signal to prompt both the tonality and the BPM
# # of the generated audio.
# res = model.generate_continuation(
#     get_bip_bip(0.125).expand(2, -1, -1), 
#     32000, ['Jazz jazz and only jazz', 
#             'Heartful EDM with beautiful synths and chords'], 
#     progress=True)
# display_audio(res, 32000)


import torchaudio
import argparse
from audiocraft.models.musicgen_shu import MusicGen
from audiocraft.data.audio import audio_write
from transformers import AutoTokenizer, AutoModelForCausalLM
from audiocraft.lib.prune_shu import prune_wanda, prune_magnitude, check_sparsity
import torch
from datasets import load_dataset


model = MusicGen.get_pretrained('facebook/musicgen-melody')

# # Print out all attributes and methods
# all_attributes = dir(model)

# # Filter out special methods and print only the regular attributes
# regular_attributes = [attr for attr in all_attributes if not attr.startswith('__')]
# for attr in regular_attributes:
#     print(attr)

model.set_generation_params(duration=8)  # generate 8 seconds.
wav = model.generate_unconditional(4)    # generates 4 unconditional audio samples
descriptions = ['classic romantic violin', 'energetic EDM', 'sad jazz']
#wav = model.generate(descriptions)          # generates 3 samples.


# melody, sr = torchaudio.load('./assets/bolero_ravel.mp3')
# generates using the melody from the given audio and the provided descriptions.
# wav = model.generate_with_chroma(descriptions, melody[None].expand(3, -1, -1), sr)



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

#model.lm(batch[0].to(device))

device = torch.device("cuda:0")
tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
prune_n, prune_m = 0, 0

#prune_magnitude(args, model.lm, tokenizer, device, prune_n=0, prune_m=0)

prune_wanda(args, model.lm, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
wav = model.generate(descriptions)              # generates 3 samples.



for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    print("shape: ", one_wav.shape)
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy= "loudness", loudness_compressor=True)


# print("*"*30)
# sparsity_ratio = check_sparsity(model)
# print(f"sparsity sanity check {sparsity_ratio:.4f}")
# print("*"*30)