from datasets import load_dataset
import os
import requests
from PIL import Image
from io import BytesIO
import backoff
import webdataset as wds
from tqdm import tqdm


# 画像をダウンロード
@backoff.on_exception(
    backoff.expo,  # 指数バックオフ
    requests.exceptions.RequestException,  # 対象例外
    max_tries=5,  # 最大リトライ回数
)
def download_image(image_url: str) -> Image:
    user_agent_string = (
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"
    )
    response = requests.get(
        image_url, headers={"User-Agent": user_agent_string}, timeout=10
    )
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image


def download_image_wrap(image_url: str) -> Image:
    try:
        return download_image(image_url)
    except Exception as e:
        print(f"Failed to process {image_url}: {e}")
        return None


def get_domain_from_question(question: str) -> str:
    for keyword, domain in domain_dict.items():
        if keyword in question:
            return domain


ds = load_dataset("line-corporation/JIC-VQA", split="train")

input_texts = []
answers = []
images = []
question_ids = []
domains = []

domain_dict = {
    "花": "jaflower30",
    "食べ物": "jafood101",
    "ランドマーク": "jalandmark10",
    "施設": "jafacility20",
}

output_dir = "dataset/jic_vqa"
os.makedirs(output_dir, exist_ok=True)
if not os.path.exists(f"{output_dir}/images.tar"):
    with wds.TarWriter(f"{output_dir}/images.tar") as sink:
        for i, example in tqdm(enumerate(ds), total=len(ds)):
            image_url = example["url"]
            image = download_image_wrap(image_url)
            # resize
            if image is not None:
                image = image.resize((224, 224))
                image = image.convert("RGB")
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

# Phase 2: Load images and populate data structures
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
# {'category': 'ガソリンスタンド', 'url': b'https://live.staticflickr.com/5536/11190751074_f97587084e_o.jpg', 'input_text': "この画像にはどの施設が映っていますか？次の四つの選択肢から正しいものを選んでください: ['スーパーマーケット', 'コンビニ', '駐車場', 'ガソリンスタンド']", 'answer': 'ガソリンスタンド', 'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=224x224 at 0x7F83A660F710>, 'question_id': '11190751074', 'domain': 'jafacility20'}
ds.to_parquet("dataset/jic_vqa.parquet")
