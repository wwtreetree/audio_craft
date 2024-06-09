from audiocraft.models import MusicGen
from audiocraft.models import MultiBandDiffusion

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

import math
import torchaudio
import torch
from audiocraft.utils.notebook import display_audio

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
display_audio(res, 32000)