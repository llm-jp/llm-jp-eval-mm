from dataclasses import dataclass


@dataclass
class GenerationConfig:
    max_new_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0
    num_beams: int = 1
    do_sample: bool = False
    use_cache: bool = True
