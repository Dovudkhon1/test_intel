#!/usr/bin/env python3

import argparse
import time
import torch
import soundfile as sf

from transformers import WhisperProcessor, WhisperForConditionalGeneration

def load_audio(path: str):
    audio, sr = sf.read(path)
    # If stereo, convert to mono
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    return audio, sr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, default="test.wav", help="Path to wav file (default: test.wav)")
    parser.add_argument("--model_id", type=str, default="openai/whisper-medium", help="HF model id")
    parser.add_argument("--language", type=str, default=None, help="Optional language hint, e.g. 'en'")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"])
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    print(f"Loading processor/model: {args.model_id}")
    processor = WhisperProcessor.from_pretrained(args.model_id)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_id)

    # If you want "no forced language/task tokens" like your snippet:
    model.config.forced_decoder_ids = None

    # Move to HPU
    if not torch.hpu.is_available():
        raise RuntimeError("torch.hpu is not available. Make sure Habana PyTorch is installed and you are on a Gaudi system.")
    device = torch.device("hpu")
    model.to(device)
    model.eval()

    # Optional: enable HPU graph (can speed up repeated runs); safe to leave off for a single test.
    # import habana_frameworks.torch.core as htcore
    # htcore.hpu_set_env()

    audio, sr = load_audio(args.audio)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")

    input_features = inputs.input_features.to(device)

    # Optional: provide language/task prompt the "Whisper way"
    # If you keep forced_decoder_ids=None, you can still provide decoder prompts manually:
    gen_kwargs = {"max_new_tokens": args.max_new_tokens}

    if args.language is not None:
        forced_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
        model.config.forced_decoder_ids = forced_ids

    # Warmup + timed run
    with torch.inference_mode():
        t0 = time.time()
        predicted_ids = model.generate(input_features, **gen_kwargs)
        # Ensure HPU is synchronized for accurate timing
        torch.hpu.synchronize()
        t1 = time.time()

    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    print("\n=== RESULTS ===")
    print(f"Audio: {args.audio}")
    print(f"Time:  {(t1 - t0):.3f} s")
    print(f"Text:  {text}")

if __name__ == "__main__":
    main()
