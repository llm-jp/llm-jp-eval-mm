import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from base_vlm import BaseVLM
from utils import GenerationConfig


class VLM(BaseVLM):
    def __init__(self, model_id: str = "CohereLabs/aya-vision-8b") -> None:
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Check for HuggingFace token
        import os
        token = os.getenv("HF_TOKEN", None)
        
        self.processor = AutoProcessor.from_pretrained(self.model_id, token=token)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, 
            device_map="auto", 
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            token=token
        )

    def generate(
        self,
        images: list[Image.Image] | None,
        text: str,
        gen_kwargs: GenerationConfig = GenerationConfig(),
    ) -> str:
        if images is None:
            images = []
        
        # Build the prompt with special tokens (based on VLLM implementation)
        # For multiple images, we need to add <image> token for each image
        image_tokens = "<image>" * len(images) if images else ""
        prompt = f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{image_tokens}{text}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
        
        # Process the inputs
        inputs = self.processor(
            text=prompt,
            images=images if images else None,
            return_tensors="pt",
            padding=True
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=gen_kwargs.max_new_tokens,
                temperature=gen_kwargs.temperature,
                top_p=gen_kwargs.top_p,
                do_sample=gen_kwargs.do_sample if gen_kwargs.temperature > 0 else False,
            )
        
        # Decode only the generated tokens
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text.strip()

    def batch_generate(
        self,
        images_list: list[list[Image.Image]] | None,
        text_list: list[str],
        gen_kwargs: GenerationConfig = GenerationConfig(),
    ) -> list[str]:
        # For batch processing, we'll process each item sequentially
        # as the model may not support true batch processing with different image counts
        results = []
        for images, text in zip(images_list or [[] for _ in text_list], text_list):
            result = self.generate(images, text, gen_kwargs)
            results.append(result)
        return results


if __name__ == "__main__":
    vlm = VLM()
    vlm.test_vlm()