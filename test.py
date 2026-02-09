import torch
import soundfile as sf
from transformers import pipeline
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

def main():
    adapt_transformers_to_gaudi()
    
    # Load audio into memory as a numpy array using soundfile
    audio_data, samplerate = sf.read("en_male.wav")
    
    # Whisper expects 16000Hz mono. Convert if necessary.
    if samplerate != 16000:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=samplerate, target_sr=16000)

    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-medium",
        device="hpu",
        torch_dtype=torch.bfloat16
    )

    # Pass the array directly instead of the filename
    result = pipe(audio_data) 
    print(result["text"])

if __name__ == "__main__":
    main()

