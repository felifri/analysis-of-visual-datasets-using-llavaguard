import asyncio
import base64
import os
from io import BytesIO
from random import uniform
import traceback
import openai
from tqdm.asyncio import tqdm

from sglang.srt.utils import kill_process_tree
from sglang_router.launch_server import launch_server
from sglang.srt.server_args import ServerArgs

def encode_image(p):
    if isinstance(p, str) or isinstance(p, os.PathLike):
        with open(p, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    elif hasattr(p, "save"):  # Check if it is a PIL image
        buffer = BytesIO()

        # TODO: Check if image has transparency, e.g. is a PNG for example
        # OSError: cannot write mode RGBA as JPEG
        # if im.mode in ("RGBA", "P"): im = im.convert("RGB")

        p.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    else:
        raise ValueError("Unsupported image format")


class LlavaGuardServer:
    @classmethod
    def setUpClass(cls, model: str = "AIML-TUDA/LlavaGuard-v1.2-7B-OV", dp_size: int = 1, port: int = 10000, is_requests_wrapper: bool = False):
        cls.model = model
        # cls.model = "AIML-TUDA/LlavaGuard-v1.2-7B-OV"
        cls.host = "127.0.0.1"
        cls.port = port
        cls.base_url = f"http://{cls.host}:{cls.port}"
        cls.api_key = "sk-123456"
        cls.is_requests_wrapper = is_requests_wrapper
        if not cls.is_requests_wrapper:
            cls.router = launch_server(
                ServerArgs(
                    model_path=cls.model,
                    host=cls.host,
                    port=cls.port,
                    api_key=cls.api_key,
                    chat_template="chatml-llava",
                    dp_size=dp_size,
                )
            )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        if not cls.is_requests_wrapper:
            cls.router.shutdown()
            # kill_process_tree(cls.router.pid, include_parent=False)

    async def request_async(self, inputs: list[dict], args={}, retries=3, timeout=300):
        """
        inputs: list of dictionaries containing keys 'prompt' and 'image'
        args: dictionary containing hyperparameters
        retries: number of retry attempts for failed requests
        timeout: timeout duration for each API call in seconds
        returns: list of completions
        """
        async with openai.AsyncOpenAI(api_key=self.api_key, base_url=self.base_url) as client:
            hyperparameters = {
                'temperature': 0.2,
                'top_p': 0.95,
                'max_tokens': 500,
            }
            hyperparameters.update(args)
            
            async def fetch_completion(input_data, attempt=1):
                base64_image = encode_image(input_data['image'])
                try:
                    response = await asyncio.wait_for(
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