import importlib

MODEL_ID_TO_CLASS_PATH = {
    "llava-hf/llava-1.5-7b-hf": "llava_1_5_7b_hf.VLM",
    "llava-hf/llava-v1.6-mistral-7b-hf": "llava_v1_6_mistral_7b_hf.VLM",
    "SakanaAI/EvoVLM-JP-v1-7B": "EvoVLM_JP_v1_7B.VLM",
    "gpt-4o-2024-05-13": "GPT_4o.VLM",
    "internlm/internlm-xcomposer2d5-7b": "internlm_xcomposer2d5_7b.VLM",
    "OpenGVLab/InternVL2-8B": "InternVL2_8B.VLM",
    "meta-llama/Llama-3.2-11B-Vision-Instruct": "Llama_3_2_11B_Vision_Instruct.VLM",
    "Kendamarron/Llama-3.2-11B-Vision-Instruct-Swallow-8B-Merge": "Llama_3_2_11B_Vision_Instruct_Swallow_8B_Merge.VLM",
    "AXCXEPT/Llama-3-EZO-VLM-1": "Llama_3_EZO_VLM_1.VLM",
    "SakanaAI/Llama-3-EvoVLM-JP-v2": "Llama_3_EvoVLM_JP_v2.VLM",
    "neulab/Pangea-7B-hf": "Pangea_7B_hf.VLM",
    "mistralai/Pixtral-12B-2409": "Pixtral_12B_2409.VLM",
    "Qwen/Qwen2-VL-7B-Instruct": "Qwen2_VL_7B_Instruct.VLM",
    "llm-jp/VILA-ja": "VILA_ja.VLM",
    "stabilityai/japanese-instructblip-alpha": "japanese_instructblip_alpha.VLM",
    "stabilityai/japanese-stable-vlm": "japanese_stable_vlm.VLM",
    "cyberagent/llava-calm2-siglip": "llava_calm2_siglip.VLM",
}


def get_class_from_path(class_path: str):
    """指定されたパスからクラスを動的にインポートして返す"""
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def get_class_from_model_id(model_id: str):
    return get_class_from_path(MODEL_ID_TO_CLASS_PATH[model_id])


if __name__ == "__main__":
    for model_id, class_path in MODEL_ID_TO_CLASS_PATH.items():
        try:
            vlm_class = get_class_from_path(class_path)
            vlm = vlm_class()
            vlm.test_vlm()
            print(f"Tested {model_id}")
        except Exception as e:
            print(f"Error testing {model_id}: {e}")
