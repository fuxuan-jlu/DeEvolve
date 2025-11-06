import http.client
import json
from openai import OpenAI

class InterfaceAPI:
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode
        self.n_trial = 5

    def get_response(self, prompt_content):
        client = OpenAI(

            api_key=self.api_key,
            base_url=self.api_endpoint,

        )
        completion = client.chat.completions.create(

            model=self.model_LLM,

            messages=[
                {"role": "user", "content": prompt_content}
            ]
        )

        return completion.choices[0].message.content