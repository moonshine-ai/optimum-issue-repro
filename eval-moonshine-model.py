import moonshine_onnx

import argparse
import os
import numpy as np
from datasets import load_dataset, Audio
from jiwer import wer, cer
from tqdm import tqdm
import torch
from transformers import pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--audio", type=str, default=None)
parser.add_argument("--language", type=str, default="en_us")
parser.add_argument("--model-type", type=str, default="base")
parser.add_argument("--models-dir", type=str, default=None)
parser.add_argument("--model-format", type=str, default="onnx")
parser.add_argument("--framework", type=str, default="onnx")
args = parser.parse_args()

language_code, country_code = args.language.split("_")

if args.framework == "onnx":
    if args.models_dir is None:
        args.models_dir = os.path.join(
            os.path.dirname(__file__), f"{language_code}-{args.model_type}"
        )
    model = moonshine_onnx.MoonshineOnnxModel(
        model_name=args.model_type,
        models_dir=args.models_dir,
        model_format=args.model_format,
    )
elif args.framework == "transformers":
    if language_code == "en":
        model_id = f"UsefulSensors/moonshine-{args.model_type}"
    else:
        model_id = f"UsefulSensors/moonshine-{args.model_type}-{language_code}"
    pipeline = pipeline(task="automatic-speech-recognition", model=model_id, device=0)
else:
    raise ValueError(f"Invalid framework: {args.framework}")

fleurs_dataset = load_dataset("google/fleurs", f"{language_code}_{country_code}")

test_dataset = fleurs_dataset["test"]

wer_total = 0
cer_total = 0
character_count = 0

for sample in tqdm(test_dataset):
    audio = sample["audio"]["array"].astype(np.float32)
    ground_truth = sample["transcription"]
    current_character_count = len(ground_truth)
    character_count += current_character_count
    if args.framework == "onnx":
        tokens = model.generate(audio.reshape(1, -1))
        transcription = moonshine_onnx.load_tokenizer().decode_batch(tokens)[0]
    elif args.framework == "transformers":
        transcription = pipeline(audio)["text"]
    print(f"Ground truth : {ground_truth}")
    print(f"Transcription: {transcription}")
    current_wer = wer(ground_truth, transcription)
    current_cer = cer(ground_truth, transcription)
    wer_total += current_wer * current_character_count
    cer_total += current_cer * current_character_count

print(f"WER: {(wer_total / character_count):.2%}")
print(f"CER: {(cer_total / character_count):.2%}")
