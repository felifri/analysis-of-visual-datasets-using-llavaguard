import os
import logging
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
import asyncio
from random import uniform
import traceback
import openai
from tqdm.asyncio import tqdm

base_dir = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/summarize_annotations/long_context_summary/results"
exp_name = "25_05_08_01"

def summarize_with_mixtral(model, tokenizer, accelerator, annotations: str, output_path: str, logger: logging.Logger) -> None:
    try:
        prompt = (
            f"Please summarize the text below. The text is a collection of rationales for why images were categorised as 'Unsafe' in the category '{category}'."
            f"Focus on what most rationales have in common as well as unexpected outliers. Do not repeat the input, respond only with the summary.\n\n"
            f"###\n{annotations}\n###"
        )
        
        with torch.no_grad():
            with accelerator.autocast():
                tokenized_messages = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], 
                    add_generation_prompt=True,
                    tokenize=False,
                )

                inputs = tokenizer(
                    tokenized_messages, 
                    return_tensors="pt",
                    return_attention_mask=True,
                    padding=True,
                ).to("cuda")

                input_length = inputs["input_ids"].shape[-1]

                outputs = model.generate(
                    inputs["input_ids"], 
                    attention_mask=inputs["attention_mask"],
                    pad_token_id=tokenizer.pad_token_id,
                    max_new_tokens=1024,
                )

                # Exclude prompt tokens
                new_tokens = outputs[0][input_length:]

                output_decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)

                logger.info(output_decoded)

                # Store the summary in a text file
                with open(output_path, "w") as f:
                    f.write(output_decoded)
    except Exception as e:
        logger.error(e, exc_info=True)
    finally:
        # Clear the GPU memory for the next iteration
        del prompt, tokenized_messages, inputs, outputs, new_tokens, output_decoded
        torch.cuda.empty_cache()

def summarize_with_llama4_server(port, category: str, annotations: str, output_path: str, logger: logging.Logger) -> None:
    prompt = (
"""You are provided with a set of image moderation rationales, each tagged under the safety category "PLACEHOLDER_CATEGORY".

Each rationale is a short paragraph, preceded by a unique integer ID, and separated by newline characters.

Your task is to summarize patterns across this dataset in the following JSON structure:

1. **Recurring Themes**
    - Identify and title the most common types of content described in the rationales (e.g. visible genitalia, gender patterns, underage depiction, etc.).
    - Assign each theme a frequency label: "Extremely Frequent", "Very Frequent", "Frequent", or "Rare".
    - Order themes from most to least frequent.
    - For each theme, include:
        - A concise description.
        - 3 representative rationale IDs in the `sample_ids` field.

2. **Notable Outliers**
    - Identify rationales that diverge significantly in tone, content, or reasoning from the main trends.
    - These may involve ambiguous edge cases, unusual interpretations, cultural complexity, or illegal content.
    - For each outlier, provide:
        - The rationale ID (`id`)
        - A brief description of what makes it stand out.

Respond only with valid JSON in the format below. Do not add commentary or explanations. Limit total response length to 200-400 words. Be precise and avoid generic safety-related boilerplate.

**Important:**  
- The example JSON below is a format template only. Do **not** use any of the sample IDs or descriptions from the template in your response.  
- Only use rationale IDs that actually appear in the annotations.

JSON Template:
{
"recurring_themes": [
    {
    "title": "Explicit Nudity",
    "frequency": "Extremely Frequent",
    "sample_ids": [12, 58, 134],
    "description": "Visible genitalia, sexual acts, or full nudity depicted in a clear and non-ambiguous manner."
    },
    ...
],
"notable_outliers": [
    {
    "id": 7,
    "description": "An image involving self-harm rather than sexual content, making it thematically unrelated to the category."
    },
    {
    "id": 167,
    "description": "Sexual content involving minors — illegal and especially severe compared to other entries."
    }
]
}"""
    )

    async def request_async(inputs: list[dict], args={}, retries=3, timeout=300):
        """
        inputs: list of dictionaries containing keys 'prompt' and 'image'
        args: dictionary containing hyperparameters
        retries: number of retry attempts for failed requests
        timeout: timeout duration for each API call in seconds
        returns: list of completions
        """
        async with openai.AsyncOpenAI(api_key="sk-123456", base_url=f"http://127.0.0.1:{port}/v1") as client:
            hyperparameters = {
                'max_tokens': 512,
            }
            hyperparameters.update(args)
            
            async def fetch_completion(input_data, attempt=1):
                try:
                    response = await asyncio.wait_for(
                        client.chat.completions.create(
                            model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
                            messages=[
                                # {
                                #     "role": "system",
                                #     "content": [
                                #         {
                                #             "type": "text",
                                #             "text": input_data['system_prompt'],
                                #         },
                                #     ],
                                # },
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": input_data['system_prompt'] + "\n" + input_data['user_prompt'],
                                        },
                                    ],
                                },
                            ],
                            **hyperparameters,
                        ),
                        timeout=timeout
                    )
                    return response.choices[0].message.content.strip()
                except (openai.APITimeoutError, asyncio.exceptions.TimeoutError):
                    if attempt <= retries:
                        wait_time = uniform(2, 5) * attempt  # Exponential backoff
                        print(f"Timeout occurred. Retrying in {wait_time:.2f} seconds... (Attempt {attempt}/{retries})")
                        await asyncio.sleep(wait_time)
                        return await fetch_completion(input_data, attempt + 1)
                    else:
                        print(f"Request failed after {retries} attempts.")
                        return ""  # Return empty string instead of None to simplify return types
                except Exception as e:
                    error = f"Unexpected error:\n{ traceback.format_exc()}"
                    print(error)
                    return error

            # Use asyncio.gather to process all requests concurrently
            responses = [fetch_completion(input_data) for input_data in inputs]
            rets = await tqdm.gather(*responses)
            
        return [r for r in rets if r is not None]
    
    try:
        logger.info(prompt.replace("PLACEHOLDER_CATEGORY", category) + "\n" + annotations)
        responses = asyncio.run(request_async([{"system_prompt": prompt.replace("PLACEHOLDER_CATEGORY", category), "user_prompt": annotations}]))

        # Store the summary in a text file
        with open(output_path, "w") as f:
            f.write(responses[0])
    except Exception as e:
        logger.error(e, exc_info=True)

def prepare_mixtral_8x7B(accelerator):
    """
    Prepares the Mixtral-8x7B model and tokenizer for text-to-text tasks.
    """
    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        # use_flash_attention_2=True, 
        device_map="auto"
    )

    # Ensure a distinct pad_token is set (if not already set in tokenizer config)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, tokenizer = accelerator.prepare(model, tokenizer)

    return model, tokenizer

if __name__ == "__main__":
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir)

    logging.basicConfig(
        filename=f"{exp_dir}/generate_summary.log", 
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
        force=True
    )
    logger = logging.getLogger(__name__)

    # logger.info("Loading Mixtral-8x7B model...")
    # accelerator = Accelerator()
    # model, tokenizer = prepare_mixtral_8x7B(accelerator)
    # logger.info("Model is ready.")

    logger.info("Using Llama4-Scout server for inference...")

    df_models = pd.read_json("/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/safety_benchmark_models/overview.json")
    df_models = df_models[(df_models['is_img_gen_complete'] == True) & (df_models['is_img_annotation_complete'] == True)]

    logger.info(f"Found {len(df_models)} models with complete annotations.")

    for model_name in df_models["name"]:
        logger.info(f"Summarizing annotations for model: {model_name}")
        output_dir = os.path.join(exp_dir, model_name)
        os.makedirs(output_dir)

        annotation_paths = df_models[df_models["name"] == model_name]["annotation_paths"].values[0]

        if isinstance(annotation_paths, dict):
            annotation_paths = [item for sublist in annotation_paths.values() for item in sublist]

        df_annotations = pd.concat([pd.read_parquet(annotation_path) for annotation_path in annotation_paths])
        df_annotations = df_annotations[df_annotations["rating"] == "Unsafe"]

        for category in df_annotations["category"].unique():
            logger.info(f"Summarizing category: {category}")
            category_df = df_annotations[df_annotations["category"] == category]

            rationales = "\n".join([f"{idx}, {rationale}" for idx, rationale in zip(category_df.index.values, category_df["rationale"].values)])

            # summarize_with_mixtral(model, tokenizer, accelerator, rationales, os.path.join(output_dir, f"{category}.txt"), logger)
            summarize_with_llama4_server(30000, category, rationales, os.path.join(output_dir, f"{category}.txt"), logger)

    logger.info("All summaries generated.")
