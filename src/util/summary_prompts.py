SUMMARISE_RATIONALES_PROMPT = (
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
    - Return a maximum of 3 outliers. In the case of more than 3, select the most unsafe ones.
    - For each outlier, provide:
        - The rationale ID (`id`)
        - A brief description of what makes it stand out.

Respond only with valid JSON in the format below. Do not add commentary or explanations. Limit total response length to 200-350 words. Be precise and avoid generic safety-related boilerplate.

**Important:**  
- The example JSON below is a format template only. Do **not** use any of the sample IDs or descriptions from the template in your response.  
- Only use rationale IDs that actually appear in the annotations.

JSON Template:
{
    "recurring_themes": [
        {
            "title": "Explicit Nudity",
            "frequency": "Extremely Frequent",
            "sample_ids": ["012", "058", "134"],
            "description": "Visible genitalia, sexual acts, or full nudity depicted in a clear and non-ambiguous manner."
        },
        ...
    ],
    "notable_outliers": [
        {
            "id": "007",
            "description": "An image involving self-harm rather than sexual content, making it thematically unrelated to the category."
        },
        {
            "id": "167",
            "description": "Sexual content involving minors — illegal and especially severe compared to other entries."
        }
    ]
}"""
).strip()

SHORTEN_CHUNKED_SUMMARIES_PROMPT = (
"""You are provided with a large JSON object that was created by combining multiple chunked summaries of the same safety category. Each chunk contains `recurring_themes` and `notable_outliers`.

Your task is to condense this combined file into a final, concise summary with the following rules:

1. **Recurring Themes**
    - Group and merge themes that are semantically similar (e.g., variations of nudity, underage exposure, or suggestive content).
    - Combine or average the frequency labels across merged items using this hierarchy:
      ["Extremely Frequent" > "Very Frequent" > "Frequent" > "Rare"]
    - Keep only the **top 5 most frequent** merged themes.
    - Choose up to 5 total representative `sample_ids` across merged themes (not per chunk).
    - Reword the theme titles and descriptions for clarity and generalization, if needed.

2. **Notable Outliers**
    - From all provided outliers, select up to **8** of the most severe, illegal, or noteworthy entries.
    - Prioritize outliers involving:
        - Illegal content (e.g. child exploitation, non-consensual acts)
        - Extreme deviation from the category
        - Unique or culturally sensitive edge cases
    - Keep their original `id` and `description`.

**Important:**
- Output must strictly follow the JSON format below.
- Do **not** invent or hallucinate entries. Use only data from the provided JSON.
- Avoid duplicates or overly similar entries.
- Be concise, neutral, and avoid generic safety-related language.
- Limit total response length to 200-500 words.

Final Output Format:
{
  "recurring_themes": [
    {
      "title": "GENERALIZED_THEME_TITLE",
      "frequency": "Averaged_Frequency_Label",
      "sample_ids": ["0123", "1456", "3789"],
      "description": "Generalized description summarizing the merged theme."
    }
  ],
  "notable_outliers": [
    {
      "id": "0999",
      "description": "Original description from source, explaining what makes it notable."
    }
  ]
}"""
).strip()