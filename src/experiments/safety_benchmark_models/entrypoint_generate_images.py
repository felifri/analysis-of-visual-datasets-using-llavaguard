import os
import logging
import pandas as pd
import subprocess
from diffusers import DiffusionPipeline

exp_name = "25_05_05_01"

def generate_images(model_name, model_dir):
    # Create directory for experiment and raise error if it already exists
    exp_dir = os.path.join(model_dir, "results", exp_name)
    os.makedirs(os.path.join(exp_dir, "images"), exist_ok=False)

    logging.basicConfig(
        filename=f"{exp_dir}/generate_images.log", 
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
        force=True
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Loading text-to-image pipeline {model_name} to GPU...")

    pipe = DiffusionPipeline.from_pretrained(model_name).to("cuda")

    logger.info("Pipeline ready.")

    prompt_file_path = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/prompt_testbench/prompt_testbench_10000_v6.csv"

    logger.info(f"Loading prompts from {prompt_file_path}...")

    prompt_df = pd.read_csv(prompt_file_path, index_col=0)

    missing_img_ids = [1547, 1548, 1549, 1550, 1551, 1553, 1554, 1555, 1556, 1558, 1559, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 1568, 1569, 1570, 1571, 1572, 1573, 1574, 1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1584, 1586, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1598, 1599, 1600, 1601, 1602, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618, 1619, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 1637, 1638, 1639, 1640, 1641, 1642, 1643, 1644, 1645, 1646, 1647, 1648, 1649, 1650, 1651, 1652, 1653, 1654, 1655, 1657, 4410, 4411, 4412, 4413, 4414, 4415, 4416, 4417, 4418, 4419, 4420, 4421, 4422, 4423, 4425, 4426, 4427, 4428, 4429, 4430, 4431, 4432, 4433, 4434, 4435, 4436, 4437, 4438, 4442, 4443, 4445, 4446, 4447, 4448, 4449, 4450, 4451, 4452, 4453, 4454, 4455, 4456, 4459, 4460, 4461, 4462, 4463, 4464, 4465, 4466, 4467, 4468, 4469, 4470, 4471, 4472, 4473, 4474, 4475, 4476, 4477, 4478, 4479, 4480, 4481, 4482, 4483, 4484, 4485, 4486, 4487, 4488, 4489, 4490, 4491, 4492, 4493, 4494, 4495, 4496, 4497, 4498, 4499, 4500, 4501, 4502, 4503, 4504, 4505, 4506, 4507, 4508, 4509, 4510, 4511, 4512, 4513, 4514, 4515, 4516, 4517, 4518, 4519, 4520, 4521, 4522, 4523, 4524, 4525, 4526, 4527, 4528, 4529, 4530, 4531, 4532, 4533, 4534, 4535, 4536, 4537, 4538, 4539, 4540, 4541, 4542, 4543, 4544, 4545, 4546, 4547, 4548, 4549, 4550, 4551, 4552, 4553, 4554, 4555, 4556, 4557, 4558, 4559, 4560, 4561, 4562, 4563, 4564, 4565, 4566, 4567, 4568, 4569, 4570, 4571, 4572, 4573, 4574, 4575, 4576, 4577, 4578, 4579, 4580, 4581, 4582, 4583, 4584, 4585, 4586, 4587, 4588, 4589, 4590, 4591, 4592, 4593, 4594, 4595, 4596, 4597, 4598, 4599, 4600, 4601, 4602, 4603, 4604, 4605, 4606, 4607, 4608, 4609, 4610, 4611, 4612, 4613, 4614, 4615, 4616, 4617, 4618, 4619, 4620, 4621, 4622, 4623, 4624, 4625, 4626, 4627, 4628, 4629, 4630, 4631, 4632, 4633, 4634, 4635, 4636, 4637, 4638, 4639, 4640, 4641, 4642, 4643, 4644, 4645, 4646, 4647, 4648, 4649, 4650, 4651, 4652, 4653, 4654, 4655, 4656, 4657, 4658, 4659, 4660, 4661, 4663, 4664, 4665, 4667, 4668, 4669, 4670, 4671, 4672, 4674, 4675, 4676, 4677, 4678, 4679, 4680, 4682, 4683, 4684, 4686, 4688, 4689, 4690, 4691, 4692, 4693, 4694, 4695, 4696, 4697, 4698, 4699, 4701]

    logger.info(f"Starting image generation: {len(missing_img_ids) if missing_img_ids else len(prompt_df)} images will be generated.")

    for idx, row in enumerate(prompt_df.itertuples()):
        if missing_img_ids:
            if idx not in missing_img_ids:
                continue

        try:            
            image = pipe(
                prompt=row.prompt,
                width=512,
                height=512
            ).images[0]
            image.save(f"{exp_dir}/images/{str(idx).zfill(len(str(len(prompt_df))))}.jpg")

            if (idx + 1) % 500 == 0 and idx > 0:
                logger.info(f"Generated {idx + 1} images.")
        except Exception as e:
            logger.error(e, exc_info=True)

    logger.info("Done generating images.")
    logger.info(f"Archiving generated images...")
    
    # Archive (without comnpression) the images and delete the original folder afterwards
    subprocess.run(
        ["tar", "-cf", f"{exp_dir}/images.tar", "-C", exp_dir, "images"],
        check=True
    )
    subprocess.run(
        ["rm", "-r", f"{exp_dir}/images"],
        check=True
    )

    logger.info(f"Done.")

if __name__ == '__main__':
    base_path = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/safety_benchmark_models"

    models = [
        # ("HiDream-ai/HiDream-I1-Full", "hidream-i1-full"),  # 12.5 hours, ⏳
        # ("stabilityai/stable-diffusion-2-1", "stable-diffusion-2_1"),  # 10 hours, ✅
        # ("stable-diffusion-v1-5/stable-diffusion-v1-5", "stable-diffusion-v1_5"),  # 11 hours, ✅
        # ("playgroundai/playground-v2.5-1024px-aesthetic", "playground-v2_5-aesthetic"),  # 25 hours, ✅
        # ("stabilityai/stable-diffusion-xl-base-1.0", "stable-diffusion-xl-base-1_0"),  # 25 hours, ✅
        # ("stablediffusionapi/newrealityxl-global-nsfw", "newrealityxl-global-nsfw"), # at least 2 days, ✅
        # ("black-forest-labs/FLUX.1-schnell", "black-forest-labs-flux-schnell"),  # 4 days, ✅
        ("Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled", "hunyuan-dit-v1_1-diffusers-distilled"), # 10.4 days, ⏳
        # ("black-forest-labs/FLUX.1-dev", "black-forest-labs-flux"),  # at least 5 days
        # ("THUDM/CogView4-6B", "cog-view-4-6b"),  # 2 days, ✅
        # ("deepseek-ai/Janus-Pro-7B", "janus-pro-7b"),  # Entry Not Found for url: https://huggingface.co/deepseek-ai/Janus-Pro-7B/resolve/main/model_index.json
    ]

    for model, model_dir in models:
        generate_images(model, os.path.join(base_path, model_dir))