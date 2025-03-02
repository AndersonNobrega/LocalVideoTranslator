import json

from llama_cpp import Llama


def llm_translation(text):
    llm = Llama(
        model_path="/home/anderson/Documents/LLM_Model/Qwen2.5-14B-Instruct-Q5_K_M.gguf",
        n_gpu_layers=20,
        n_ctx=2048,
        verbose=False,
    )

    prompt = f"""
    Translate the following text from portuguese to brazillian portuguese: {text}.
    
    Output the translation in the following JSON format:
    
    {{
        "translations": [
            "translated_text1", "translated_text2"
        ]
    }}

    
    Review your output to make sure it follows the specific output structure.
    Try to identify nuances in the original text and correctly translate it to brazillian language.
    """

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are an specialist in translating texts from any languages to brazillian portuguese.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.5,
        top_p=0.95,
    )

    return json.loads(response["choices"][0]["message"]["content"])
