import os
import math
import logging
import pandas as pd
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

base_dir = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/summarize_annotations/long_context_summary/results"
exp_name = "25_05_20_03"

system_instruction = (
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

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    reraise=True
)
def generate_summary(client: genai.Client, category: str, rationales: str) -> str:
    """
    Generate a summary using the Google GenAI client.
    
    Args:
        category (str): The category for which to generate the summary.
        rationales (str): The rationales to summarize.
    
    Returns:
        str: The generated summary.
    """
    response = client.models.generate_content(
        model="gemini-2.5-pro-preview-05-06",
        contents=[system_instruction.replace("PLACEHOLDER_CATEGORY", category), rationales],
        config=types.GenerateContentConfig(
            safety_settings=[
                types.SafetySetting(
                    category=cat,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ) for cat in [types.HarmCategory.HARM_CATEGORY_HARASSMENT, types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY]
            ]
        )
    )

    if not response.text:
        logger.error(f"No response text received from the model: {response}")
        return ""
    
    return response.text

if __name__ == "__main__":
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir)

    logging.basicConfig(
        filename=f"{exp_dir}/generate_summary.log", 
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
        force=True
    )

    logging.getLogger("google_genai.models").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)

    logger.info("Creating Google GenAI client...")
    project_id = os.getenv("GOOGLE_GENAI_PROJECT_ID")

    client = genai.Client(
        vertexai=True,
        project=project_id,
        location="us-central1",
        # http_options=types.HttpOptions(api_version="v1"),
    )
    logger.info("Client ready.")

    df_datasets = pd.read_json('/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/datasets/datasets.json')
    df_models = pd.read_json('/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/safety_benchmark_models/overview.json')

    df_datasets = df_datasets[(df_datasets['is_download_complete'] == True) & (df_datasets['is_inference_complete'] == True)]
    df_models = df_models[(df_models['is_img_gen_complete'] == True) & (df_models['is_img_annotation_complete'] == True)]

    df = df_datasets
    df = df[df["name"].isin(["CelebA", "ImageNet"])]

    logger.info(f"Summarising annotations for: {df['name'].values}")

    for name in df["name"]:
        logger.info(f"Summarizing annotations for model: {name}")
        output_dir = os.path.join(exp_dir, name)
        os.makedirs(output_dir)

        annotation_paths = df[df["name"] == name]["annotation_paths"].values[0]

        if isinstance(annotation_paths, dict):
            annotation_paths = [item for sublist in annotation_paths.values() for item in sublist]

        df_annotations = pd.concat([pd.read_parquet(annotation_path) for annotation_path in annotation_paths])
        df_annotations = df_annotations[df_annotations["rating"] == "Unsafe"]

        for category in df_annotations["category"].unique():
            category_df = df_annotations[df_annotations["category"] == category]
            logger.info(f"Summarizing category {category} with {len(category_df)} rationales.")

            total_rationales = len(category_df)
            max_chunk_size = 2050

            if total_rationales == 0:
                logger.warning(f"No rationales found for category {category}. Skipping.")
                continue

            if total_rationales > max_chunk_size:
                num_chunks = math.ceil(total_rationales / max_chunk_size)
                chunk_size = math.ceil(total_rationales / num_chunks)
                logger.info(f"Category {category} has {total_rationales} rationales and will be split into {num_chunks} chunks of size ~{chunk_size}.")

                for idx_chunk in range(num_chunks):
                    start_idx = idx_chunk * chunk_size
                    end_idx = min(start_idx + chunk_size, total_rationales)
                    logger.info(f"Processing chunk {idx_chunk+1} of {num_chunks} for category {category}. Rationales {start_idx} to {end_idx}.")
                    chunk_df = category_df.iloc[start_idx:end_idx]
                    rationales = "Rationales:\n\n" + "\n".join(
                        [f"{idx}, {rationale}" for idx, rationale in zip(chunk_df.index.values, chunk_df["rationale"].values)]
                    )
                    try:
                        summary = generate_summary(client, category, rationales)
                        if summary:
                            filename = f"{category}_{idx_chunk+1}_of_{num_chunks}.txt"
                            with open(os.path.join(output_dir, filename), "w") as f:
                                f.write(summary)
                    except Exception as e:
                        logger.error(e, exc_info=True)

                continue  # Skip the rest of the loop for chunked categories

            # If the number of rationales is less than the max chunk size, process them all at once
            rationales = "Rationales:\n\n" + "\n".join([f"{idx}, {rationale}" for idx, rationale in zip(category_df.index.values, category_df["rationale"].values)])

            try:
                summary = generate_summary(client, category, rationales)

                # Store the summary in a text file
                # TODO: Extract the JSON from the response and save it in a structured format.
                if summary:
                    with open(os.path.join(output_dir, f"{category}.txt"), "w") as f:
                        f.write(summary)
            except Exception as e:
                logger.error(e, exc_info=True)

    logger.info("All summaries generated.")
