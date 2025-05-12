import importlib

MODEL_ID_TO_CLASS_PATH = {
    "llava-hf/llava-1.5-7b-hf": "llava_1_5.VLM",
    "llava-hf/llava-1.5-13b-hf": "llava_1_5.VLM",
    "llava-hf/llava-v1.6-mistral-7b-hf": "llava_1_6_mistral_hf.VLM",
    "SakanaAI/EvoVLM-JP-v1-7B": "evovlm_jp_v1.VLM",
    "gpt-4o-2024-05-13": "gpt4o.VLM",
    "gpt-4o-2024-11-20": "gpt4o.VLM",
    "internlm/internlm-xcomposer2d5-7b": "xcomposer2d5.VLM",
    "OpenGVLab/InternVL2-8B": "internvl2.VLM",
    "OpenGVLab/InternVL2-26B": "internvl2.VLM",
    "meta-llama/Llama-3.2-11B-Vision-Instruct": "llama_3_2_vision.VLM",
    "meta-llama/Llama-3.2-90B-Vision-Instruct": "llama_3_2_vision.VLM",
    "Kendamarron/Llama-3.2-11B-Vision-Instruct-Swallow-8B-Merge": "llama_3_2_vision.VLM",
    "AXCXEPT/Llama-3-EZO-VLM-1": "llama_3_evovlm_jp_v2.VLM",
    "SakanaAI/Llama-3-EvoVLM-JP-v2": "llama_3_evovlm_jp_v2.VLM",
    "neulab/Pangea-7B-hf": "pangea_hf.VLM",
    "mistralai/Pixtral-12B-2409": "pixtral.VLM",
    "Qwen/Qwen2-VL-2B-Instruct": "qwen2_vl.VLM",
    "Qwen/Qwen2-VL-7B-Instruct": "qwen2_vl.VLM",
    "Qwen/Qwen2-VL-72B-Instruct": "qwen2_vl.VLM",
    "Qwen/Qwen2.5-VL-3B-Instruct": "qwen2_5_vl.VLM",
    "Qwen/Qwen2.5-VL-7B-Instruct": "qwen2_5_vl.VLM",
    "Qwen/Qwen2.5-VL-32B-Instruct": "qwen2_5_vl.VLM",
    "Qwen/Qwen2.5-VL-72B-Instruct": "qwen2_5_vl.VLM",
    "llm-jp/llm-jp-3-vila-14b": "llm_jp_3_vila.VLM",
    "stabilityai/japanese-instructblip-alpha": "japanese_instructblip_alpha.VLM",
    "stabilityai/japanese-stable-vlm": "japanese_stable_vlm.VLM",
    "cyberagent/llava-calm2-siglip": "llava_calm2_siglip.VLM",
    "Efficient-Large-Model/VILA1.5-13b": "vila.VLM",
    "google/gemma-3-1b-it": "gemma3.VLM",
    "google/gemma-3-4b-it": "gemma3.VLM",
    "google/gemma-3-12b-it": "gemma3.VLM",
    "google/gemma-3-27b-it": "gemma3.VLM",
    "sbintuitions/sarashina2-vision-8b": "sarashina2_vision.VLM",
    "sbintuitions/sarashina2-vision-14b": "sarashina2_vision.VLM",
    "microsoft/Phi-4-multimodal-instruct": "phi4_multimodal.VLM",
    "MIL-UT/Asagi-14B": "asagi.VLM",
    "turing-motors/Heron-NVILA-Lite-1B": "heron_nvila.VLM",
    "turing-motors/Heron-NVILA-Lite-2B": "heron_nvila.VLM",
    "turing-motors/Heron-NVILA-Lite-15B": "heron_nvila.VLM",
    "turing-motors/Heron-NVILA-Lite-33B": "heron_nvila.VLM",
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
            vlm = vlm_class(model_id)
            vlm.test_vlm()
            print(f"Tested {model_id}")
        except Exception as e:
            print(f"Error testing {model_id}: {e}")
