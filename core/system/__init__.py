from .base import *
from .ddpm import *
from .encoder import *
from .vae import VAESystem


systems = {
    'encoder': EncoderSystem,
    'ddpm': DDPM,
    'vae': VAESystem,
}