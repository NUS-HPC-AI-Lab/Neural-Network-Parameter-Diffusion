from .base import *
from .ddpm import *
from .encoder import *


systems = {
    'encoder': EncoderSystem,
    'ddpm': DDPM,
}