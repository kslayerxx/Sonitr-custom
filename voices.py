# voices.py

import torch
from soni_translate.logging_setup import logger  
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid, 
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,  
)
from vci_pipeline import VC
import traceback, pdb
from lib.audio import load_audio
import numpy as np  
import os, shutil
from fairseq import checkpoint_utils
from app_rvc import apply_shift, change_tempo, change_speed
import soundfile as sf
from gtts import gTTS
import edge_tts
import asyncio
from soni_translate.utils import remove_directory_contents, create_directories


class ClassVoices:

    def __init__(self):
        self.file_index = "" # root

    def apply_conf(self, f0method, 
                   model_voice_path00, transpose00, file_index2_00,  
                   model_voice_path01, transpose01, file_index2_01,
                   model_voice_path02, transpose02, file_index2_02,
                   model_voice_path03, transpose03, file_index2_03,
                   model_voice_path04, transpose04, file_index2_04,
                   model_voice_path05, transpose05, file_index2_05,
                   model_voice_path99, transpose99, file_index2_99):

        # Implementation of apply_conf method
    
    def custom_voice(self, args):
        # Implementation of custom_voice method

    def make_test(self, args):
        # Implementation of make_test method

    def __call__(self, speakers_list, audio_files):
       # Implementation of __call__ method

