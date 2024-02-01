from .base import *
from .ddpm import *
from .encoder import *
from .vae import VAESystem
from .ae_ddpm import AE_DDPM
from .explore import Explore

systems = {
    'encoder': EncoderSystem,
    'ddpm': DDPM,
    'vae': VAESystem,
    'ae_ddpm': AE_DDPM,
    'explore': Explore,
}