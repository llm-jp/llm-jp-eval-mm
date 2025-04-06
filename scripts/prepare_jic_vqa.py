from datasets import load_dataset
import os
import requests
from PIL import Image as PILImage
from io import BytesIO
import backoff
import webdataset as wds
from tqdm import tqdm
from typing import List, Optional, Dict


# 画像をダウンロード
@backoff.on_exception(
    backoff.expo,
    requests.exceptions.RequestException,
    max_tries=5,
)
def download_image(image_url: str) -> PILImage.Image:
    user_agent_string = (
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"
    )
    response = requests.get(
        image_url, headers={"User-Agent": user_agent_string}, timeout=10
    )
    response.raise_for_status()
    image = PILImage.open(BytesIO(response.content)).convert("RGB")
    return image


def download_image_wrap(image_url: str) -> Optional[PILImage.Image]:
    try:
        return download_image(image_url)
    except Exception as e:
        print(f"Failed to process {image_url}: {e}")
        return None


def get_domain_from_question(question: str) -> Optional[str]:
    for keyword, domain in domain_dict.items():
        if keyword in question:
            return domain
    return None


# 型アノテーション付き変数定義
input_texts: List[str] = []
answers: List[str] = []
images: List[Optional[PILImage.Image]] = []
question_ids: List[str] = []
domains: List[str] = []

# ドメイン辞書
domain_dict: Dict[str, str] = {
    "花": "jaflower30",
    "食べ物": "jafood101",
    "ランドマーク": "jalandmark10",
    "施設": "jafacility20",
}

# データセット読み込み＆画像保存処理
output_dir = "dataset/jic_vqa"
os.makedirs(output_dir, exist_ok=True)

ds = load_dataset("line-corporation/JIC-VQA", split="train")

if not os.path.exists(f"{output_dir}/images.tar"):
    with wds.TarWriter(f"{output_dir}/images.tar") as sink:
        for i, example in tqdm(enumerate(ds), total=len(ds)):
            image_url = example["url"]
            image = download_image_wrap(image_url)
            if image is not None:
                image = image.resize((224, 224)).convert("RGB")
            if image is None:
                continue
            sample = {
                "__key__": str(example["id"]),
                "jpg": image,
                "txt": example["category"],
                "url.txt": image_url,
                "question.txt": example["question"],
            }
            sink.write(sample)

# WebDatasetの読み込みと加工処理
ds = load_dataset("webdataset", data_files=f"{output_dir}/images.tar", split="train")
print(ds)
print(ds[0])

ds = ds.remove_columns(["__url__"])
ds = ds.rename_columns(
    {
        "txt": "category",
        "url.txt": "url",
        "question.txt": "question",
    }
)

ds = ds.map(
    lambda x: {
        "input_text": x["question"].decode("utf-8"),
        "url": x["url"].decode("utf-8").encode("utf-8"),
        "answer": str(x["category"]),
        "image": x["jpg"],
        "question_id": int(x["__key__"]),
        "domain": get_domain_from_question(str(x["question"].decode("utf-8"))),
    }
)

ds = ds.remove_columns(["question", "__key__", "jpg"])

print(ds)
print(ds[0])
ds.to_parquet("dataset/jic_vqa.parquet")
