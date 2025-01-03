import io
import os
import unicodedata
import string
import glob

import torch
import random

ALL_LETTERS = string.ascii_letters
N_LETTERS = len(ALL_LETTERS)

def unicode_to_ascii(s) :
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_LETTERS
    )
