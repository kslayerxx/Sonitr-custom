import librosa
import tempfile
import ffmpeg
import os
from pydub import AudioSegment

def apply_shift(entries, shift):
    """Shifts start and end times in entries by specified amount"""
    for entry in entries:
        entry['start_time'] += shift
        entry['end_time'] += shift
        
def change_tempo(audio_path, speed):
    """Changes tempo/speed of audio at given path"""
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
    
    ffmpeg_cmd = (ffmpeg
                  .input(audio_path)
                  .filter('atempo', speed)
                  .output(temp_file)
                  .overwrite_output()
                 ) 
    
    ffmpeg_cmd.run()
    
    os.remove(audio_path)
    os.rename(temp_file, audio_path)

def change_speed(audio_path, speed):
    """Changes speed of audio while maintaining pitch"""
    y = numpy.linspace(0, 1, librosa.time_to_samples(librosa.get_duration(filename=audio_path)))
    y_fast = librosa.effects.time_stretch(y, speed)

    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name 
    
    librosa.output.write_wav(temp_file, y_fast, samplerate=22050)
    
    os.remove(audio_path)
    os.rename(temp_file, audio_path)
    
def overlay_audio(base_audio, overlay_audio, position):
    """Overlays overlay_audio onto base_audio at given position"""
    base_audio = base_audio.overlay(overlay_audio, position=position)
    return base_audio

def export_audio(audio, export_path):
    """Exports AudioSegment to file at export_path"""
    audio.export(export_path, format="wav")
    
def create_silence(duration):
    """Creates a silent audio segment of specified duration"""
    return AudioSegment.silent(duration=duration)
