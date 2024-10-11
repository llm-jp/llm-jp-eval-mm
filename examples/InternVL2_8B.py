import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from typing import Union

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# 画像の数だけ画像を読み込んでcatする
def load_images(images: Union[Image.Image, list[Image.Image]]):
    if isinstance(images, list):
        tuples = ()
        for image in images:
            tuples += (load_image(image).to(torch.bfloat16).cuda(),)
        return torch.cat(tuples, dim=0)
    else:
        return load_image(images).to(torch.bfloat16).cuda()


# 画像の数だけ <image> をpromptの先頭に追加する
def add_image_tags(images: Union[Image.Image, list[Image.Image]], prompt: str) -> str:
    if isinstance(images, list):
        num_images = len(images)
    else:
        num_images = 1

    image_tags = "<image> " * num_images
    new_prompt = image_tags + prompt

    return new_prompt


class VLM:
    model_id = "OpenGVLab/InternVL2-8B"

    def __init__(self) -> None:
        self.model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True, use_fast=False
        )

    def generate(self, image, text: str, max_new_tokens: int = 256):
        text = text.replace("<image>", "")
        if "<image>" not in text:
            if isinstance(image, list):
                image_tokens = ["<image>"] * len(image)
                image_tokens = " ".join(image_tokens)
                text = f"{image_tokens}\n{text}"
            else:
                text = f"<image>\n{text}"
        if isinstance(image, list):
            pixel_values_list = []
            for img in image:
                pixel_values = (
                    load_image(img, max_num=12)
                    .to(self.model.device)
                    .to(self.model.dtype)
                )
                pixel_values_list.append(pixel_values)
            num_patches_list = [
                pixel_values.size(0) for pixel_values in pixel_values_list
            ]
            pixel_values = torch.cat(pixel_values_list, dim=0)

        else:
            num_patches_list = None
            pixel_values = (
                load_image(image, max_num=12).to(self.model.device).to(self.model.dtype)
            )

        generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)

        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            text,
            generation_config,
            num_patches_list=num_patches_list,
        )
        generated_text = response
        return generated_text


if __name__ == "__main__":
    import requests
    from PIL import Image

    model = VLM()
    image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_file, stream=True).raw)
    print(model.generate(image, "What is in the image?"))
    print(
        model.generate([image, image], "What is the difference between these images?")
    )
