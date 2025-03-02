import json

from ollama import chat
from pydantic import BaseModel


class TextTranslations(BaseModel):
    translations: list[str]


def llm_translation(text):
    output = """
    {
        "translations": [
            "translated_text1", 
            "translated_text2"
        ]
    }
    """

    prompt = f"""
    Translate the following text from **Portuguese** to **Brazilian Portuguese**:

    {text}

    ### Output Format:
    Ensure the translation follows this **JSON structure**:
    {output}

    ### Output Schema:
    {TextTranslations.model_json_schema()}

    ### Guidelines:
    - Maintain the original meaning while adapting to **Brazilian Portuguese** nuances.
    - Use **natural and fluent** language suitable for Brazilian speakers.
    - **Strictly** adhere to the specified output format.
    - Double-check the output to ensure it follows the required JSON structure.

    Return only the JSON output—no additional explanations.
    """

    response = chat(
        model="qwen2.5:latest",
        messages=[
            {
                "role": "system",
                "content": "You are an specialist in translating texts from any languages to brazillian portuguese.",
            },
            {"role": "user", "content": prompt},
        ],
        format=TextTranslations.model_json_schema(),
        options={"temperature": 0.5, "top_p": 0.95},
    )

    return json.loads(response["message"]["content"])
