import asyncio
import logging
import os
import numpy as np
import pandas as pd
import rtpt
from experiments.safety_token_logprobs.sglang_gpt_server import LlavaGuardServer
from util.annotation_utils import save_json_annotations
from util.file_utils import get_file_paths
from util.policy import POLICY_DEFAULT

image_dir = "/pfss/mlde/workspaces/mlde_wsp_Shared_Datasets/laion2B-en/test_10000/data_336"
output_dir = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/safety_token_logprobs/results_7B_laion2B-en_test_10000_336_25_02_25_01"


def main():
    # Try to create the output dir. If it already exists, throw an error to avoid overwriting data.
    os.makedirs(output_dir)
    
    logging.basicConfig(
        filename=f"{output_dir}/log.txt", 
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s'
    )
    logger = logging.getLogger(__name__)

    server = LlavaGuardServer()

    image_paths, image_names = get_file_paths(image_dir, file_extension='.jpg')
    inputs = [
        {
            'image': path,
            'image_name': name,
            'prompt': POLICY_DEFAULT,
        } for path, name in zip(image_paths, image_names)
    ]

    # server.setUpClass(model="AIML-TUDA/LlavaGuard-v1.2-0.5B-OV")
    server.setUpClass(model="AIML-TUDA/LlavaGuard-v1.2-7B-OV")

    batch_size = 5
    num_batches = str(len(inputs)//batch_size + 1).zfill(len(str(len(inputs)//batch_size)))

    logger.info(f"Total {len(inputs)} annotations & token logprobs will be generated in {num_batches} batches")
    try:
        rt = rtpt.RTPT(name_initials='FG', experiment_name=f'LlavaGuard inference /w sglang', max_iterations=len(inputs)//batch_size + 1)
        rt.start()

        for i in range(0, len(inputs), batch_size):
            idx_batch = str(i // batch_size).zfill(len(str(len(inputs)//batch_size)))
            logger.info(f"Running batch {idx_batch}/{num_batches}")

            annotations, logprobs = asyncio.run(server.request_async([{"image": input_['image'], "prompt": input_['prompt']} for input_ in inputs[i:i+batch_size]]))
            # annotations = asyncio.run(server.request_async([{"image": input_['image'], "prompt": input_['prompt']} for input_ in inputs[i:i+batch_size]]))

            # logger.info(f"Saving annotations of batch {idx_batch}/{num_batches}.")
            invalid_json = save_json_annotations(annotations, f"{output_dir}/image_annotations/batch_{idx_batch}", [input_['image_name'] for input_ in inputs[i:i+batch_size]])

            if invalid_json:
                logger.warning(f"Invalid JSON annotations found in batch {idx_batch}/{num_batches}: {invalid_json}")

            logger.info(f"Saving token logprobs of batch {idx_batch}/{num_batches}.")
            df = pd.DataFrame()

            for top_two_logprobs, image_name in zip(logprobs, [input_['image_name'] for input_ in inputs[i:i+batch_size]]):
                df = pd.concat([df, pd.DataFrame(
                    index=[image_name],
                    data={
                        'index': image_name,
                        'logprob_1_token': top_two_logprobs[0].token,
                        'logprob_1_logprob': top_two_logprobs[0].logprob,
                        'logprob_1_linprob': f"{np.round(np.exp(top_two_logprobs[0].logprob) * 100, 2)} %",
                        'logprob_2_token': top_two_logprobs[1].token,
                        'logprob_2_logprob': top_two_logprobs[1].logprob,
                        'logprob_2_linprob': f"{np.round(np.exp(top_two_logprobs[1].logprob) * 100, 2)} %",
                    }
                )])
                
            # Save the dataframe to a CSV file
            os.makedirs(f"{output_dir}/token_logprobs", exist_ok=True)
            df.to_csv(f"{output_dir}/token_logprobs/batch_{idx_batch}.csv", index=False, index_label="index")

            rt.step()
    except Exception as e:
        logger.error(e, exc_info=True)
    finally:
        logger.info("Shutting down the server.")
        server.tearDownClass()

if __name__ == '__main__':
    main()
