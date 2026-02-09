import torch
import time
import habana_frameworks.torch.core as htcore
from transformers import pipeline
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

def main():
    # 1. Adapt transformers for Intel Gaudi HPU
    adapt_transformers_to_gaudi()
    
    # 2. Configure device and model
    # We use 'medium' for a balance of speed and accuracy on a single HPU
    model_id = "openai/whisper-medium"
    device = "hpu"
    
    print(f"Loading {model_id} onto {device}...")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=device,
        torch_dtype=torch.bfloat16 # Optimized for Gaudi 2
    )

    # 3. Transcribe audio
    audio_file = "en_male_sample.wav"
    print(f"Transcribing {audio_file}...")
    
    start_time = time.time()
    result = pipe(audio_file, generate_kwargs={"language": "english"})
    
    # Synchronize to get accurate timing on HPU
    torch.hpu.synchronize() 
    end_time = time.time()

    print("\n" + "="*20)
    print("TRANSCRIPTION:")
    print(result["text"])
    print("="*20)
    print(f"Inference Time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
