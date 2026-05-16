import base64

import openai
from tqdm.asyncio import tqdm

# from sglang.srt.utils import kill_child_process
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)

def encode_image(p):
    with open(p, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


class LlavaGuardServer:
    @classmethod
    def setUpClass(cls, model="AIML-TUDA/LlavaGuard-v1.2-7B-OV"):
        cls.model = model
        cls.base_url = "http://127.0.0.1:10000"
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--chat-template",
                "chatml-llava",
                # "--dp",
                # "4",
                # "--log-requests",
            ],
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        # kill_child_process(cls.process.pid, include_self=True)
        kill_process_tree(cls.process.pid, include_parent=False)

    async def request_async(self, inputs: list[dict], args={}):
        """
        inputs: list of dictionaries containing keys 'prompt' and 'image'
        args: dictionary containing hyperparameters
        returns: list of completions
        """
        
        async with openai.AsyncOpenAI(api_key=self.api_key, base_url=self.base_url) as client:
            hyperparameters = {
                'temperature': 0.2,
                'top_p': 0.95,
                'max_tokens': 500,
                'logprobs': True,
                'top_logprobs': 2,
            }
            hyperparameters.update(args)
            response = []
            for input_data in inputs:
                base64_image = encode_image(input_data['image'])
                response.append(
                    client.chat.completions.create(
                        model="default",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        },
                                    },
                                    {
                                        "type": "text",
                                        "text": input_data['prompt'],
                                    },
                                ],
                            },
                        ],
                        **hyperparameters,
                    )
                )
            rets = await tqdm.gather(*response)
        # client.close()
        # await client.close()  # Ensure the client is properly closed
        return [r.choices[0].message.content.strip() for r in rets]  , [r.choices[0].logprobs.content[6].top_logprobs[:2] for r in rets]