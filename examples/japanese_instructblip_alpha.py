# TODO: 複数画像のレスポンスが空白

from PIL import Image, ImageDraw, ImageFont
from transformers import LlamaTokenizer, AutoModelForVision2Seq, BlipImageProcessor
from typing import Union
from base_vlm import BaseVLM
from utils import GenerationConfig


# helper function to format input prompts
def build_prompt(prompt="", sep="\n\n### "):
    sys_msg = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。"
    p = sys_msg
    roles = ["指示", "応答"]
    user_query = "与えられた画像について、詳細に述べてください。"
    msgs = [": \n" + user_query, ": "]
    if prompt:
        roles.insert(1, "入力")
        msgs.insert(1, ": \n" + prompt)
    for role, msg in zip(roles, msgs):
        p += sep + role + msg
    return p


def add_order_label(image, label, font_size=40):
    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Define font for the label
    # font_path = fm.findfont(fm.FontProperties(family=font_family))
    # font_path = os.path.join(__file__, os.pardir, "arial.ttf")
    # font = ImageFont.truetype(font_path, font_size)
    font = ImageFont.load_default()

    # Calculate text size and position
    text_width = text_height = font_size
    label_background_margin = 10
    label_background_size = (
        text_width + 2 * label_background_margin,
        text_height + 2 * label_background_margin,
    )

    # Draw a solid white square for the label background
    label_background_position = (0, 0)  # Top-left corner
    draw.rectangle(
        [
            label_background_position,
            (
                label_background_position[0] + label_background_size[0],
                label_background_position[1] + label_background_size[1],
            ),
        ],
        fill="white",
    )

    # Add the label text in black over the white square
    label_position = (label_background_margin, label_background_margin)
    draw.text(label_position, label, font=font, fill="black")

    return image


def resize_image_height(image, fixed_size):
    # Resize image, maintaining aspect ratio
    width, height = image.size
    new_size = int(width * fixed_size / height), fixed_size
    return image.resize(new_size, Image.Resampling.LANCZOS)


def concatenate_images_horizontal(image_list):
    # Concatenate images horizontally
    widths, heights = zip(*(i.size for i in image_list))
    total_width = sum(widths)
    max_height = max(heights)
    assert all(height == max_height for height in heights)
    new_im = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for im in image_list:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return new_im


def resize_image_width(image, fixed_size):
    # Resize image, maintaining aspect ratio
    width, height = image.size
    new_size = fixed_size, int(height * fixed_size / width)
    return image.resize(new_size, Image.Resampling.LANCZOS)


def concatenate_images_vertical(image_list):
    # Concatenate images horizontally
    widths, heights = zip(*(i.size for i in image_list))
    total_height = sum(heights)
    max_width = max(widths)
    assert all(width == max_width for width in widths)
    new_im = Image.new("RGB", (max_width, total_height))
    y_offset = 0
    for im in image_list:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
    return new_im


def process_images_horizontal(original_images, size):
    images = []
    for i, img in enumerate(original_images):
        # Resize image
        img_resized = resize_image_height(img, fixed_size=size)

        # Add order label
        img_labeled = add_order_label(img_resized, f"[{i+1}]")

        # Append to list
        images.append(img_labeled)

    # Concatenate all images
    return concatenate_images_horizontal(images)


def process_images_vertical(original_images, size):
    images = []
    for i, img in enumerate(original_images):
        # Resize image
        img_resized = resize_image_width(img, fixed_size=size)

        # Add order label
        img_labeled = add_order_label(img_resized, f"[{i+1}]")

        # Append to list
        images.append(img_labeled)

    # Concatenate all images
    return concatenate_images_vertical(images)


def process_images(images, size=1008):
    concat_horizontal = process_images_horizontal(images, size)
    concat_vertical = process_images_vertical(images, size)

    hw, hh = concat_horizontal.size
    vw, vh = concat_vertical.size

    ha = hw / hh
    va = vh / vw

    if ha > va:
        return concat_vertical
    else:
        return concat_horizontal


class VLM(BaseVLM):
    model_id = "stabilityai/japanese-instructblip-alpha"

    def __init__(self) -> None:
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        ).to("cuda")
        self.processor = BlipImageProcessor.from_pretrained(self.model_id)
        self.tokenizer = LlamaTokenizer.from_pretrained(
            "novelai/nerdstash-tokenizer-v1", additional_special_tokens=["▁▁"]
        )

    def generate(
        self,
        images: Union[Image.Image, list[Image.Image]],
        text: str,
        gen_kwargs: GenerationConfig = GenerationConfig(),
    ):
        text = text.replace("<image>", "")
        prompt = build_prompt(prompt=text)
        if isinstance(images, list):
            images = [process_images(images)]
        inputs = self.processor(images=images, return_tensors="pt", truncation=True)
        text_encoding = self.tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        )
        text_encoding["qformer_input_ids"] = text_encoding["input_ids"].clone()
        text_encoding["qformer_attention_mask"] = text_encoding[
            "attention_mask"
        ].clone()
        inputs.update(text_encoding)

        # autoregressively complete prompt
        output = self.model.generate(
            **inputs.to(self.model.device),
            **gen_kwargs.__dict__,
        )
        # TODO: white space return problem some times
        response = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        generated_text = response[0].strip()
        return generated_text


if __name__ == "__main__":
    vlm = VLM()
    vlm.test_vlm()
