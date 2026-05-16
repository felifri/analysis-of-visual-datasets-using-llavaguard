"""Unit test for LlamaGuard-3-11B-Vision: verify image is actually processed."""

import torch
import numpy as np
from PIL import Image


def main():
    from transformers import AutoModelForVision2Seq, AutoProcessor

    model_id = "meta-llama/Llama-Guard-3-11B-Vision"
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, local_files_only=True,
    ).to("cuda").eval()

    # Create test images
    white = Image.new("RGB", (512, 512), (255, 255, 255))
    # Red image (should be safe)
    red = Image.new("RGB", (512, 512), (255, 0, 0))

    print("=" * 70)
    print("TEST 1: Verify image tokens are present in input")
    print("=" * 70)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image."},
            ],
        },
    ]

    input_prompt = processor.apply_chat_template(conversation, return_tensors="pt")
    print(f"Chat template output (type): {type(input_prompt)}")
    if isinstance(input_prompt, str):
        print(f"Chat template output: {input_prompt[:500]}")
        print(f"Contains <image> token: {'<image>' in input_prompt}")
    elif isinstance(input_prompt, list):
        print(f"Chat template output: {input_prompt[0][:500] if input_prompt else 'empty'}")

    inputs = processor(text=input_prompt, images=white, return_tensors="pt").to("cuda")
    print(f"\ninput_ids shape: {inputs['input_ids'].shape}")
    print(f"pixel_values present: {'pixel_values' in inputs}")
    if "pixel_values" in inputs:
        print(f"pixel_values shape: {inputs['pixel_values'].shape}")
    else:
        print("WARNING: No pixel_values — image is NOT being processed!")

    # Check if there are image tokens in the input
    decoded = processor.decode(inputs["input_ids"][0], skip_special_tokens=False)
    print(f"\nDecoded prompt (first 500 chars):\n{decoded[:500]}")

    print("\n" + "=" * 70)
    print("TEST 2: Compare outputs WITH image vs WITHOUT image")
    print("=" * 70)

    # With image
    prompt_len = len(inputs["input_ids"][0])
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=0)
    response_with = processor.decode(outputs[0, prompt_len:], skip_special_tokens=True).strip()
    print(f"With white image: '{response_with}'")

    # With red image
    inputs_red = processor(text=input_prompt, images=red, return_tensors="pt").to("cuda")
    prompt_len_red = len(inputs_red["input_ids"][0])
    with torch.no_grad():
        outputs_red = model.generate(**inputs_red, max_new_tokens=20, do_sample=False, pad_token_id=0)
    response_red = processor.decode(outputs_red[0, prompt_len_red:], skip_special_tokens=True).strip()
    print(f"With red image: '{response_red}'")

    # Without image (text-only)
    conversation_text_only = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
            ],
        },
    ]
    input_prompt_text = processor.apply_chat_template(conversation_text_only, return_tensors="pt")
    inputs_text = processor(text=input_prompt_text, return_tensors="pt").to("cuda")
    prompt_len_text = len(inputs_text["input_ids"][0])
    with torch.no_grad():
        outputs_text = model.generate(**inputs_text, max_new_tokens=20, do_sample=False, pad_token_id=0)
    response_text = processor.decode(outputs_text[0, prompt_len_text:], skip_special_tokens=True).strip()
    print(f"Without image (text only): '{response_text}'")

    print(f"\nSame response with/without image: {response_with == response_text}")
    if response_with == response_text:
        print("WARNING: Image appears to NOT be influencing the output!")

    print("\n" + "=" * 70)
    print("TEST 3: Check a few actual generated images from tar")
    print("=" * 70)

    import tarfile, os, random
    tar_path = "<your folder>"
    if os.path.exists(tar_path):
        with tarfile.open(tar_path, "r:") as tf:
            members = sorted([m for m in tf.getmembers() if m.isfile() and m.name.endswith(".jpg")], key=lambda m: m.name)
            # Pick first 5 images
            for member in members[:5]:
                f = tf.extractfile(member)
                if f:
                    img = Image.open(f).convert("RGB")
                    inputs = processor(text=input_prompt, images=img, return_tensors="pt").to("cuda")
                    prompt_len = len(inputs["input_ids"][0])
                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=0)
                    response = processor.decode(outputs[0, prompt_len:], skip_special_tokens=True).strip()
                    print(f"  {os.path.basename(member.name)}: '{response}'")
    else:
        print(f"  Tar not found: {tar_path}")

    print("\n" + "=" * 70)
    print("TEST 4: Check 10 images LlavaGuard rated UNSAFE (from multiple conditions)")
    print("=" * 70)

    import pandas as pd

    # Load LlavaGuard annotations to find unsafe images
    annot_base = "<your folder>"
    tar_base = "<your folder>"

    # Sample unsafe images from different conditions/models
    sources = [
        ("dose_C1", os.path.join(annot_base, "dose_C1.parquet"), os.path.join(tar_base, "dose_response/C1/images.tar")),
        ("dose_C2", os.path.join(annot_base, "dose_C2.parquet"), os.path.join(tar_base, "dose_response/C2/images.tar")),
        ("dose_C0", os.path.join(annot_base, "dose_C0.parquet"), os.path.join(tar_base, "dose_response/C0/images.tar")),
        ("dose_C6", os.path.join(annot_base, "dose_C6.parquet"), os.path.join(tar_base, "dose_response/C6/images.tar")),
        ("prx-512-sft", os.path.join(annot_base, "prx-512-sft.parquet"), os.path.join(tar_base, "prx_existing/prx-512-sft/images.tar")),
    ]

    unsafe_tested = 0
    unsafe_caught = 0
    safe_tested = 0
    safe_caught = 0

    for source_name, parquet_path, tar_path in sources:
        if not os.path.exists(parquet_path) or not os.path.exists(tar_path):
            print(f"  Skipping {source_name}: missing files")
            continue

        df = pd.read_parquet(parquet_path)
        unsafe_ids = df[df["rating"] == "Unsafe"].index.tolist()
        safe_ids = df[df["rating"] == "Safe"].index.tolist()

        if not unsafe_ids:
            print(f"  Skipping {source_name}: no unsafe images")
            continue

        # Pick 2 unsafe + 2 safe from each source
        import random
        rng = random.Random(42)
        sample_unsafe = rng.sample(unsafe_ids, min(2, len(unsafe_ids)))
        sample_safe = rng.sample(safe_ids, min(2, len(safe_ids)))

        # Load images from tar
        with tarfile.open(tar_path, "r:") as tf:
            members = {os.path.basename(m.name).removesuffix(".jpg"): m
                       for m in tf.getmembers() if m.isfile() and m.name.endswith(".jpg")}

            for img_id in sample_unsafe + sample_safe:
                is_unsafe_llavaguard = img_id in sample_unsafe
                llavaguard_rating = "Unsafe" if is_unsafe_llavaguard else "Safe"
                llavaguard_cat = df.loc[img_id, "category"] if "category" in df.columns else "?"

                if img_id not in members:
                    continue

                f = tf.extractfile(members[img_id])
                if not f:
                    continue
                img = Image.open(f).convert("RGB")

                inputs = processor(text=input_prompt, images=img, return_tensors="pt").to("cuda")
                prompt_len = len(inputs["input_ids"][0])
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=0)
                response = processor.decode(outputs[0, prompt_len:], skip_special_tokens=True).strip()

                lines = response.strip().split("\n")
                lg3_rating = "Safe" if lines[0].lower().startswith("safe") else "Unsafe"
                lg3_cats = lines[1] if len(lines) > 1 and lg3_rating == "Unsafe" else ""

                agree = "AGREE" if lg3_rating == llavaguard_rating else "DISAGREE"
                print(f"  {source_name}/{img_id}: LlavaGuard={llavaguard_rating}({llavaguard_cat}) LG3={lg3_rating}({lg3_cats}) [{agree}]")

                if is_unsafe_llavaguard:
                    unsafe_tested += 1
                    if lg3_rating == "Unsafe":
                        unsafe_caught += 1
                else:
                    safe_tested += 1
                    if lg3_rating == "Safe":
                        safe_caught += 1

    print(f"\n  Unsafe images caught by LG3: {unsafe_caught}/{unsafe_tested}")
    print(f"  Safe images confirmed by LG3: {safe_caught}/{safe_tested}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("If with/without image give same response, the image is being ignored.")
    print("If pixel_values is missing from inputs, the processor isn't handling images.")
    print("If LG3 catches 0 unsafe images that LlavaGuard flagged, image understanding is broken.")


if __name__ == "__main__":
    main()
