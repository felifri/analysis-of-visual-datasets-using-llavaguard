"""Unit test for ShieldGemma-2: verify lm_head weight tying fix and policy prompt behavior."""

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, ShieldGemma2ForImageClassification


def load_model():
    model_id = "google/shieldgemma-2-4b-it"
    model = ShieldGemma2ForImageClassification.from_pretrained(model_id, local_files_only=True)
    model.model.lm_head.weight = model.model.get_input_embeddings().weight
    model = model.eval().to("cuda")
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    return model, processor


def make_test_images():
    images = {}
    images["solid_white"] = Image.new("RGB", (512, 512), (255, 255, 255))
    images["solid_blue"] = Image.new("RGB", (512, 512), (0, 100, 200))
    arr = np.zeros((512, 512, 3), dtype=np.uint8)
    for y in range(512):
        arr[y, :, 1] = int(100 + 155 * y / 512)
        arr[y, :, 2] = int(200 - 150 * y / 512)
    images["green_gradient"] = Image.fromarray(arr)
    return images


def main():
    model, processor = load_model()
    images = make_test_images()
    policies = ["dangerous", "sexual", "violence"]

    # --- Test 1: Without explicit policy arg ---
    print("=" * 70)
    print("TEST 1: processor(images=...) — NO explicit policies arg")
    print("=" * 70)
    for name, img in images.items():
        inputs = processor(images=[img], return_tensors="pt").to("cuda")
        decoded = processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
        has_policy = any(p in decoded.lower() for p in ["dangerous", "sexual", "violence", "policy"])
        print(f"\n--- {name} ---")
        print(f"  input_ids shape: {inputs['input_ids'].shape}")
        print(f"  Has policy text in prompt: {has_policy}")

        with torch.inference_mode():
            scores = model(**inputs)
        probs = scores.probabilities.cpu().numpy()
        print(f"  probs shape: {probs.shape}")
        for i in range(len(probs)):
            p = probs[i, 0]
            print(f"  Row {i}: P(violates)={p:.4f} -> {'UNSAFE' if p > 0.5 else 'safe'}")

    # --- Test 2: With explicit policies ---
    print("\n" + "=" * 70)
    print("TEST 2: processor(images=..., policies=[...]) — WITH policies")
    print("=" * 70)
    for name, img in images.items():
        inputs = processor(images=[img], return_tensors="pt", policies=policies).to("cuda")
        print(f"\n--- {name} ---")
        print(f"  input_ids shape: {inputs['input_ids'].shape}")

        with torch.inference_mode():
            scores = model(**inputs)
        probs = scores.probabilities.cpu().numpy()
        for k, pol in enumerate(policies):
            if k < len(probs):
                p = probs[k, 0]
                print(f"  {pol}: P(violates)={p:.4f} -> {'UNSAFE' if p > 0.5 else 'safe'}")

    # --- Test 3: Batch without policies ---
    print("\n" + "=" * 70)
    print("TEST 3: Batch of 3 images — NO policies")
    print("=" * 70)
    img_list = list(images.values())
    name_list = list(images.keys())
    inputs = processor(images=img_list, return_tensors="pt").to("cuda")
    print(f"input_ids shape: {inputs['input_ids'].shape}")

    with torch.inference_mode():
        scores = model(**inputs)
    probs = scores.probabilities.cpu().numpy()
    print(f"probs shape: {probs.shape} (expected ({len(img_list)*3}, 2))")

    for j, name in enumerate(name_list):
        print(f"\n  {name}:")
        for k, pol in enumerate(policies):
            idx = j * 3 + k
            if idx < len(probs):
                p = probs[idx, 0]
                print(f"    {pol}: P(violates)={p:.4f} -> {'UNSAFE' if p > 0.5 else 'safe'}")

    # --- Test 4: Batch WITH policies ---
    print("\n" + "=" * 70)
    print("TEST 4: Batch of 3 images — WITH policies")
    print("=" * 70)
    inputs = processor(images=img_list, return_tensors="pt", policies=policies).to("cuda")
    print(f"input_ids shape: {inputs['input_ids'].shape}")

    with torch.inference_mode():
        scores = model(**inputs)
    probs = scores.probabilities.cpu().numpy()
    print(f"probs shape: {probs.shape}")

    for j, name in enumerate(name_list):
        print(f"\n  {name}:")
        for k, pol in enumerate(policies):
            idx = j * 3 + k
            if idx < len(probs):
                p = probs[idx, 0]
                print(f"    {pol}: P(violates)={p:.4f} -> {'UNSAFE' if p > 0.5 else 'safe'}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("If solid white/blue/gradient are flagged UNSAFE, something is wrong.")
    print("Compare Test 1 vs 2 to see if policy prompts matter.")
    print("Compare Test 3 vs 4 to see if batching + policies interact.")


if __name__ == "__main__":
    main()
