from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import requests
import torch
import pandas as pd
from util.policy import POLICY_DEFAULT

"""
Inference Timings:
10911.9599609375ms
11361.2978515625ms
10510.3125ms
10191.396484375ms
11244.61328125ms
10150.0244140625ms
11195.9912109375ms
10524.5625ms
10793.0224609375ms
9291.130859375ms
==> 10617.43115234375ms avg
"""

model_500m = LlavaForConditionalGeneration.from_pretrained('AIML-TUDA/LlavaGuard-v1.2-0.5B-OV-hf')
processor_500m = AutoProcessor.from_pretrained('AIML-TUDA/LlavaGuard-v1.2-0.5B-OV-hf')

model_7b = LlavaForConditionalGeneration.from_pretrained('AIML-TUDA/LlavaGuard-v1.2-7B-OV-hf')
processor_7b = AutoProcessor.from_pretrained('AIML-TUDA/LlavaGuard-v1.2-7B-OV-hf')

model_13b = LlavaForConditionalGeneration.from_pretrained('AIML-TUDA/LlavaGuard-v1.1-13B-hf')
processor_13b = AutoProcessor.from_pretrained('AIML-TUDA/LlavaGuard-v1.1-13B-hf')

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": POLICY_DEFAULT},
            ],
    },
]

hyperparameters = {
    "max_new_tokens": 200,
    "do_sample": True,
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 50,
    "num_beams": 2,
    "use_cache": True,
    # "return_dict_in_generate": True,  # Only return json
}

img_urls = [
    "https://fastly.picsum.photos/id/662/1300/876.jpg?hmac=MtaSJNSKD6c3MIvW5T98S_KrE8bRbXKUpMSCJsMilW0",
    "https://fastly.picsum.photos/id/293/1300/876.jpg?hmac=Hq-z2-F6TnxA1sGIwvNzYeshjRBnCZ8eiSbSEU59Vwo",
    "https://fastly.picsum.photos/id/876/1300/876.jpg?hmac=WpaYlQYR68X8Z6asB1VAH3M4ji79FCWfZxi33nQ052g",
    "https://fastly.picsum.photos/id/938/1300/876.jpg?hmac=e2sNTtrCaKdtb1PWLPmcfx8NkzQIOjqAxZ3CiwfdgFg",
    "https://fastly.picsum.photos/id/157/1300/876.jpg?hmac=D-aRGmHtntB6uJ-X77dt5zFrJSL8l2mIA9drErCmdU8",
    "https://fastly.picsum.photos/id/48/1300/876.jpg?hmac=8p2dbt69Y6lc3202Zn13q6TOBu5OsyJAlzZi9KKur6w",
    "https://fastly.picsum.photos/id/127/1300/876.jpg?hmac=vfhFxKFvzPxqYnbX_ve84wDWp5uHZ_oV0qAluZETIm8",
    "https://fastly.picsum.photos/id/1050/1300/876.jpg?hmac=IzFCsxh0rOusbXDWzEhCpbud2rst36RBmI6S_5Hl4j8",
    "https://fastly.picsum.photos/id/1062/1300/876.jpg?hmac=9PTif3A5Ds6k2_Asm6NIGTckEfxr9YmDqAanmNaPzUc",
    "https://fastly.picsum.photos/id/605/1300/876.jpg?hmac=z9yYR6AJk05nfUp7PWrE8iZmiIzSQQcH76bNhKCQIz8"
]

# https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch
timings = {}
start_events = [torch.cuda.Event(enable_timing=True) for _ in range(len(img_urls))]
end_events = [torch.cuda.Event(enable_timing=True) for _ in range(len(img_urls))]

# TODO: Use rtpt to rename process dynamically
# TODO: Use tqdm to show progress bar
for name, model, processor in [("500M", model_500m, processor_500m), ("7B", model_7b, processor_7b) , ("13B", model_13b, processor_13b)]:
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    model.to('cuda:0')

    for idx, url in enumerate(img_urls):
        image = Image.open(requests.get(url, stream=True).raw)

        inputs = processor(text=text_prompt, images=image, return_tensors="pt")
        inputs = {k: v.to('cuda:0') for k, v in inputs.items()}

        start_events[idx].record()

        # Generate
        output = model.generate(**inputs, **hyperparameters)

        end_events[idx].record()

        decoded = processor.decode(output[0], skip_special_tokens=True)
        print(f"Image {idx + 1}/{len(img_urls)} processed.")
    
    torch.cuda.synchronize()
    times = [start.elapsed_time(end) for start, end in zip(start_events, end_events)]
    timings[f"inference {name} [ms]"] = times
    print("Inference Timings:")
    [print(f"{time}ms") for time in times]
    print(f"==> {sum(times) / len(times)}ms avg")

df = pd.DataFrame(timings)
print(df)

results_path = '/pfss/mlde/workspaces/mlde_swp_KIServiceCenter/finngu/LlavaGuard/src/experiments/perf_benchmark_hf/results.csv'
df.to_csv(results_path, encoding='utf-8', index=False, header=True)
