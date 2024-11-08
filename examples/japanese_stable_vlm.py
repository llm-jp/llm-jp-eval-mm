import torch
from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoImageProcessor
from PIL import Image, ImageDraw, ImageFont
from base_vlm import BaseVLM
from utils import GenerationConfig

# helper function to format input prompts
TASK2INSTRUCTION = {
    "caption": "画像を詳細に述べてください。",
    "tag": "与えられた単語を使って、画像を詳細に述べてください。",
    "vqa": "与えられた画像を下に、質問に答えてください。",
}


def build_prompt(task="caption", input=None, sep="\n\n### "):
    assert (
        task in TASK2INSTRUCTION
    ), f"Please choose from {list(TASK2INSTRUCTION.keys())}"
    if task in ["tag", "vqa"]:
        assert input is not None, "Please fill in `input`!"
        if task == "tag" and isinstance(input, list):
            input = "、".join(input)
    else:
        assert input is None, f"`{task}` mode doesn't support to input questions"
    sys_msg = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。"
    p = sys_msg
    roles = ["指示", "応答"]
    instruction = TASK2INSTRUCTION[task]
    msgs = [": \n" + instruction, ": \n"]
    if input:
        roles.insert(1, "入力")
        msgs.insert(1, ": \n" + input)
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
    model_id = "stabilityai/japanese-stable-vlm"

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model.to(self.device)

    def generate(
        self, images, text: str, gen_kwargs: GenerationConfig = GenerationConfig()
    ):
        # instruct blip does not expect the <image> tag
        text = text.replace("<image>", "")
        prompt = build_prompt(task="vqa", input=text)
        if isinstance(images, list):
            images = [process_images(images)]
        inputs = self.processor(images=images, return_tensors="pt", truncation=True)
        text_encoding = self.tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        )
        inputs.update(text_encoding)

        # autoregressively complete prompt
        output = self.model.generate(
            **inputs.to(self.device, dtype=self.model.dtype), **gen_kwargs.__dict__
        )[0]

        generated_text = self.tokenizer.decode(output, skip_special_tokens=True).strip()
        return generated_text


if __name__ == "__main__":
    vlm = VLM()
    vlm.test_vlm()
