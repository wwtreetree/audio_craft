from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy
from audiocraft.data.audio import audio_write

import torchaudio
import argparse
from audiocraft.models.musicgen_shu import MusicGen
from audiocraft.data.audio import audio_write
from transformers import AutoTokenizer, AutoModelForCausalLM
from audiocraft.lib.prune_shu import prune_wanda, prune_magnitude, check_sparsity
import torch
from audiocraft.models import MusicGen
from audiocraft.models import MultiBandDiffusion



USE_DIFFUSION_DECODER = False
# Using small model, better results would be obtained with `medium` or `large`.
model = MusicGen.get_pretrained('facebook/musicgen-small')
if USE_DIFFUSION_DECODER:
    mbd = MultiBandDiffusion.get_mbd_musicgen()


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='facebook/musicgen-small', help='LLaMA model')
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

prune_wanda(args, model.lm, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
#model.lm.transformer.layers


outputs = model.generate(
    descriptions = [
        #'80s pop track with bassy drums and synth',
        #'90s rock song with loud guitars and heavy drums',
        'Progressive rock drum and bass solo',
        #'Punk Rock song with loud drum and power guitar',
        #'Bluesy guitar instrumental with soulful licks and a driving rhythm section',
        #'Jazz Funk song with slap bass and powerful saxophone',
        'drum and bass beat with intense percussions'
    ],
    progress=True, return_tokens=True
)

# Loop through the outputs and save each one to a separate file
for i, output_waveform in enumerate(outputs[0]):
    output_waveform = output_waveform.cpu()
    output_file_path = f"./output/text_{i+1}.wav"
    torchaudio.save(output_file_path, output_waveform, sample_rate=32000)
    print(f"Audio saved to {output_file_path}")


if USE_DIFFUSION_DECODER:
    out_diffusion = mbd.tokens_to_wav(outputs[1])
    for i, diffusion_waveform in enumerate(out_diffusion):
        diffusion_waveform = diffusion_waveform.cpu()
        diffusion_output_file_path = f"text_{i+1}.wav"
        torchaudio.save(diffusion_output_file_path, diffusion_waveform, sample_rate=32000)
        print(f"Diffusion audio saved to {diffusion_output_file_path}")





# display_audio(output[0], sample_rate=32000)
# if USE_DIFFUSION_DECODER:
#     out_diffusion = mbd.tokens_to_wav(output[1])
#     display_audio(out_diffusion, sample_rate=32000)