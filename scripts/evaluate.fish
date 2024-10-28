#!/usr/bin/env fish

# model list: Select from here
set -l models \
    # "EvoVLM_JP_v1_7B" \
    # "InternVL2_8B"
    # "japanese_instructblip_alpha"\
    # "japanese_stable_vlm" \
    # "Llama_3_2_11B_Vision_Instruct"\
    # "llava_1_5_7b_hf"\
    # "llava_calm2_siglip"\
    # "llava_v1_6_mistral_7b_hf"
    # "Qwen2_VL_7B_Instruct"
    "GPT_4o"

# task list: Select from here
set -l tasks \
    # "japanese-heron-bench" \
    # "ja-vg-vqa-500" \
    # "ja-vlm-bench-in-the-wild" \
    # "ja-multi-image-vqa" 
    # "jmmmu"
    "jdocqa"

# execute evaluation
for model in $models
    for task in $tasks
        echo "Execute: rye run python examples/sample.py --class_path $model --task_id $task"
        rye run python examples/sample.py --class_path $model --task_id $task
    end
end

echo "Evaluation is done."