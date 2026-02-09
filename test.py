import torch
import soundfile as sf

# Enable Gaudi optimizations BEFORE loading models
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
adapt_transformers_to_gaudi()

from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

AUDIO_FILE = "en_male.wav"
MODEL_NAME = "openai/whisper-medium"

# Load audio (must be 16kHz mono)
audio, sr = sf.read(AUDIO_FILE)
if sr != 16000:
    raise ValueError("Audio must be 16kHz")

# Load processor & model
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
).to("hpu")

model.eval()

# Preprocess
inputs = processor(
    audio,
    sampling_rate=16000,
    return_tensors="pt",
)

input_features = inputs.input_features.to("hpu")

# Force English transcription
forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="english",
    task="transcribe"
)

# Inference
with torch.no_grad(), torch.autocast(device_type="hpu", dtype=torch.bfloat16):
    predicted_ids = model.generate(
        input_features,
        forced_decoder_ids=forced_decoder_ids,
        max_new_tokens=256,
    )

# Decode
text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print("\nTranscription:\n", text)
