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
from audio_utils import apply_shift, change_tempo, change_speed
import soundfile as sf
from gtts import gTTS
import edge_tts
import asyncio
from soni_translate.utils import remove_directory_contents, create_directories

# call inference
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

        #self.filename = filename
        self.f0method = f0method # pm
        
        self.model_voice_path00 = model_voice_path00
        self.transpose00 = transpose00
        self.file_index200 = file_index2_00

        self.model_voice_path01 = model_voice_path01
        self.transpose01 = transpose01
        self.file_index201 = file_index2_01

        self.model_voice_path02 = model_voice_path02
        self.transpose02 = transpose02
        self.file_index202 = file_index2_02

        self.model_voice_path03 = model_voice_path03
        self.transpose03 = transpose03
        self.file_index203 = file_index2_03

        self.model_voice_path04 = model_voice_path04
        self.transpose04 = transpose04
        self.file_index204 = file_index2_04

        self.model_voice_path05 = model_voice_path05
        self.transpose05 = transpose05
        self.file_index205 = file_index2_05

        self.model_voice_path99 = model_voice_path99
        self.transpose99 = transpose99
        self.file_index299 = file_index2_99
        return "CONFIGURATION APPLIED"

    def custom_voice(self,
        _values, # filter indices
        audio_files, # all audio files
        model_voice_path='',
        transpose=0,
        f0method='pm',
        file_index='',
        file_index2='',
        tempo_mode='',  
        tempo_speed='',
        shift_mode='',
        speed='',
        ):

        #hubert_model = None

        generate_inference(
            sid=model_voice_path,  # model path
            to_return_protect0=0.33,
            to_return_protect1=0.33
        )

        for _value_item in _values:
            filename = audio_files[_value_item] if _value_item != "test" else audio_files[0]
            #filename = audio_files[_value_item]
            try:
                logger.info(f"{audio_files[_value_item]}, {model_voice_path}")
            except:
                pass

            info_, (sample_, audio_output_) = vc_single(
                sid=0,
                input_audio_path=filename, # Original file
                f0_up_key=transpose, # transpose for m to f and reverse 0 12
                f0_file=None,
                f0_method= f0method,
                file_index= file_index, # dir pwd?
                file_index2= file_index2,
                # file_big_npy1,
                index_rate= float(0.66),
                filter_radius= int(3),
                resample_sr= int(0),
                rms_mix_rate= float(0.25),
                protect= float(0.33),
            )

            sf.write(
                file= filename, # Overwrite
                samplerate=sample_,
                data=audio_output_
            )

        # detele the model

    def make_test(self, 
        tts_text, 
        tts_voice, 
        model_path,
        index_path,
        transpose,
        f0_method,
        ):

        create_directories("test")
        remove_directory_contents("test")
        filename = "test/test.wav"

        if "SET_LIMIT" == os.getenv("DEMO"):
          if len(tts_text) > 60:
            tts_text = tts_text[:60]
            logger.warning("DEMO; limit to 60 characters")

        language = tts_voice[:2]
        try:
          asyncio.run(edge_tts.Communicate(tts_text, "-".join(tts_voice.split('-')[:-1])).save(filename))
        except:
          try:
              tts = gTTS(tts_text, lang=language)
              tts.save(filename)
              tts.save
              logger.warning(f'No audio was received. Please change the tts voice for {tts_voice}. USING gTTS.')
          except:
            tts = gTTS('a', lang=language)
            tts.save(filename)
            logger.error('Audio will be replaced.')

        shutil.copy("test/test.wav", "test/real_test.wav")

        self([],[]) # start modules

        self.custom_voice(
            ["test"], # filter indices
            ["test/test.wav"], # all audio files
            model_voice_path=model_path,
            transpose=transpose,
            f0method=f0_method,
            file_index='',
            file_index2=index_path,
        )
        return "test/test.wav", "test/real_test.wav"

    def __call__(self, speakers_list, audio_files):

        speakers_indices = {}

        for index, speak_ in enumerate(speakers_list):
            if speak_ in speakers_indices:
                speakers_indices[speak_].append(index)
            else:
                speakers_indices[speak_] = [index]

        
        # find models and index
        global weight_root, index_root, config, hubert_model
        weight_root = "weights"
        names = []
        for name in os.listdir(weight_root):
            if name.endswith(".pth"):
                names.append(name)

        index_root = "logs"
        index_paths = []
        for name in os.listdir(index_root):
            if name.endswith(".index"):
                index_paths.append(name)

        logger.info(f"{names}, {index_paths}")
        # config machine
        hubert_model = None
        config = Config('cuda:0', is_half=True) # config = Config('cpu', is_half=False) # cpu

        # filter by speaker
        for _speak, _values in speakers_indices.items():
            logger.debug(f"{_speak}, {_values}")
            #for _value_item in _values:
            #  self.filename = "audio2/"+audio_files[_value_item]
            ###print(audio_files[_value_item])

            #vc(_speak, _values, audio_files)

            if _speak == "SPEAKER_00":
              self.custom_voice(
                    _values, # filteredd
                    audio_files,
                    model_voice_path=self.model_voice_path00,
                    file_index2=self.file_index200,
                    transpose=self.transpose00,
                    f0method=self.f0method,
                    file_index=self.file_index,
                    )
            elif _speak == "SPEAKER_01":
                self.custom_voice(
                    _values,
                    audio_files,
                    model_voice_path=self.model_voice_path01,
                    file_index2=self.file_index201,
                    transpose=self.transpose01,
                    f0method=self.f0method,
                    file_index=self.file_index,
                )
            elif _speak == "SPEAKER_02":
                self.custom_voice(
                    _values,
                    audio_files,
                    model_voice_path=self.model_voice_path02,
                    file_index2=self.file_index202,
                    transpose=self.transpose02,
                    f0method=self.f0method,
                    file_index=self.file_index,
                )
            elif _speak == "SPEAKER_03":
                self.custom_voice(
                    _values,
                    audio_files,
                    model_voice_path=self.model_voice_path03,
                    file_index2=self.file_index203,
                    transpose=self.transpose03,
                    f0method=self.f0method,
                    file_index=self.file_index,
                )
            elif _speak == "SPEAKER_04":
                self.custom_voice(
                    _values,
                    audio_files,
                    model_voice_path=self.model_voice_path04,
                    file_index2=self.file_index204,
                    transpose=self.transpose04,
                    f0method=self.f0method,
                    file_index=self.file_index,
                )
            elif _speak == "SPEAKER_05":
                self.custom_voice(
                    _values,
                    audio_files,
                    model_voice_path=self.model_voice_path05,
                    file_index2=self.file_index205,
                    transpose=self.transpose05,
                    f0method=self.f0method,
                    file_index=self.file_index,
                )
            elif _speak == "SPEAKER_99":
                self.custom_voice(
                    _values,
                    audio_files,
                    model_voice_path=self.model_voice_path99,
                    file_index2=self.file_index299,
                    transpose=self.transpose99,
                    f0method=self.f0method,
                    file_index=self.file_index,
                )
            else:
                pass
