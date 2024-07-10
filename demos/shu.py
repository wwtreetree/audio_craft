from transformers import MusicgenForConditionalGeneration
import scipy
import torchaudio
import argparse
from audiocraft.models.musicgen_shu import MusicGen
from audiocraft.data.audio import audio_write
from transformers import AutoTokenizer, AutoModelForCausalLM
from audiocraft.lib.prune_shu import prune_wanda, prune_magnitude, check_sparsity
import torch
from datasets import load_dataset


model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
unconditional_inputs = model.get_unconditional_inputs(num_samples=1)
audio_values = model.generate(**unconditional_inputs, do_sample=True, max_new_tokens=256)


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
sampling_rate = model.config.audio_encoder.sampling_rate
scipy.io.wavfile.write("musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())