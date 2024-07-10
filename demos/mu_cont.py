import math
import torchaudio
import torch
from audiocraft.utils.notebook import display_audio
import torchaudio
import argparse
from audiocraft.models.musicgen_shu import MusicGen
from audiocraft.data.audio import audio_write
from transformers import AutoTokenizer, AutoModelForCausalLM
from audiocraft.lib.prune_shu import prune_wanda, prune_magnitude, check_sparsity
import torch
from audiocraft.models import MusicGen
from audiocraft.models import MultiBandDiffusion
import scipy


USE_DIFFUSION_DECODER = False

# Using small model, better results would be obtained with `medium` or `large`.
model = MusicGen.get_pretrained('facebook/musicgen-small')
if USE_DIFFUSION_DECODER:
    mbd = MultiBandDiffusion.get_mbd_musicgen()

model.set_generation_params(
    use_sampling=True,
    top_k=250,
    duration=30
)


def get_bip_bip(bip_duration=0.125, frequency=440,
                duration=0.5, sample_rate=32000, device="cuda"):
    """Generates a series of bip bip at the given frequency."""
    t = torch.arange(
        int(duration * sample_rate), device="cuda", dtype=torch.float) / sample_rate
    wav = torch.cos(2 * math.pi * 440 * t)[None]
    tp = (t % (2 * bip_duration)) / (2 * bip_duration)
    envelope = (tp >= 0.5).float()
    return wav * envelope

# Here we use a synthetic signal to prompt both the tonality and the BPM
# of the generated audio.
res = model.generate_continuation(
    get_bip_bip(0.125).expand(2, -1, -1), 
    32000, ['Jazz jazz and only jazz', 
            'Heartful EDM with beautiful synths and chords'], 
    progress=True)
#display_audio(res, 32000)

for i, output_waveform in enumerate(res):
    output_waveform = output_waveform.cpu()
    output_file_path = f"./output/res_{i+1}.wav"
    torchaudio.save(output_file_path, output_waveform, sample_rate=32000)
    print(f"Audio saved to {output_file_path}")



#scipy.io.wavfile.write("res.wav", rate=3200, data=res.cpu().numpy())

# You can also use any audio from a file. Make sure to trim the file if it is too long!
prompt_waveform, prompt_sr = torchaudio.load("./assets/bach.mp3")
prompt_duration = 2
prompt_waveform = prompt_waveform[..., :int(prompt_duration * prompt_sr)]
output = model.generate_continuation(prompt_waveform, prompt_sample_rate=prompt_sr, progress=True, return_tokens=True)
#display_audio(output[0], sample_rate=32000)

for i, output_waveform in enumerate(output[0]):
    output_waveform = output_waveform.cpu()
    output_file_path = f"./output/cont_{i+1}.wav"
    torchaudio.save(output_file_path, output_waveform, sample_rate=32000)
    print(f"Audio saved to {output_file_path}")



#scipy.io.wavfile.write("output_0.wav", rate=3200, data=output[0].cpu().numpy())


if USE_DIFFUSION_DECODER:
    out_diffusion = mbd.tokens_to_wav(output[1])
    display_audio(out_diffusion, sample_rate=32000)