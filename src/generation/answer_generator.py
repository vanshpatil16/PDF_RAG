import os
from typing import List, Dict
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

class QwenAPIGenerator:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.model_id = model_id
        self.client = InferenceClient(
            model=self.model_id,
            token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
        )

    def generate_response(self, messages: List[Dict[str, str]], max_new_tokens: int = 512) -> str:
        """
        Generates a response using the Hugging Face Inference API.
        """
        response = ""
        for message in self.client.chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            stream=True,
        ):
            delta = message.choices[0].delta.content
            if delta:
                response += delta
        return response

if __name__ == "__main__":
    generator = QwenAPIGenerator()
    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    print(generator.generate_response(messages))
