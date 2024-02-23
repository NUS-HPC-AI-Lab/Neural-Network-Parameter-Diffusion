from .base import *
from .ddpm import *
from .encoder import *
from .vae import VAESystem
from .ae_ddpm import AE_DDPM

systems = {
    'encoder': EncoderSystem,
    'ddpm': DDPM,
    'vae': VAESystem,
    'ae_ddpm': AE_DDPM,
}