import argparse
import time
import torch
import soundfile as sf

from transformers import WhisperProcessor, WhisperForConditionalGeneration


def load_audio(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:  # stereo → mono
        audio = audio.mean(axis=1)
    return audio, sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", type=str, help="Path to .wav audio file")
    args = parser.parse_args()

    # --- Fixed configuration ---
    MODEL_ID = "openai/whisper-medium"
    LANGUAGE = "en"
    TASK = "transcribe"
    MAX_NEW_TOKENS = 256

    # --- Sanity check ---
    if not torch.hpu.is_available():
        raise RuntimeError("HPU not available. Check Habana / SynapseAI setup.")

    device = torch.device("hpu")

    print("Loading Whisper processor and model...")
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

    # Force English transcription
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=LANGUAGE,
        task=TASK
    )

    model.to(device)
    model.eval()

    # --- Load audio ---
    audio, sr = load_audio(args.audio_path)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    # --- Inference ---
    print("Running inference on HPU...")
    with torch.inference_mode():
        start = time.time()
        predicted_ids = model.generate(
            input_features,
            max_new_tokens=MAX_NEW_TOKENS
        )
        torch.hpu.synchronize()
        end = time.time()

    text = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

    print("\n=== TRANSCRIPTION ===")
    print(text)
    print(f"\nInference time: {end - start:.3f}s")


if __name__ == "__main__":
    main()
