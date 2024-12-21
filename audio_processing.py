import numpy as np
import torch
from pydub import AudioSegment
import librosa
import time 
def load_audio(audio_file):
    samples = AudioSegment.from_file(audio_file)
    sample_rate = samples.frame_rate
    target_sr = 16000
    samples = np.array(samples.get_array_of_samples())
    samples = convert_samples_to_float32(samples)
    if target_sr != sample_rate:
        samples = librosa.core.resample(samples, orig_sr=sample_rate, target_sr=target_sr)
    
    features = torch.tensor(samples, dtype=torch.float)
    f, fl = features, torch.tensor(features.shape[0]).long()
    f, fl = f.unsqueeze(0), fl.unsqueeze(0)
    return f, fl

def convert_samples_to_float32(samples):
    """Convert sample type to float32."""
    float32_samples = samples.astype('float32')
    if np.issubdtype(samples.dtype, np.integer): 
        bits = np.iinfo(samples.dtype).bits
        float32_samples *= 1.0 / 2 ** (bits - 1)
    return float32_samples

def process_audio(file_path: str, model, language_code: str):
    f, f_l = load_audio(file_path)

    # ASR inference
    if model is None:
        return "Model not loaded"
    with torch.no_grad():
        start = time.time()
        output = model.forward(f, f_l, language_code)
        end = time.time()
        print(start-end)
    return output
