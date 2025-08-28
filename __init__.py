# /ComfyUI/custom_nodes/LLM_Manga_Director/__init__.py

from .nodes import LLM_Manga_Director

NODE_CLASS_MAPPINGS = {
    "LLM_Manga_Director": LLM_Manga_Director,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLM_Manga_Director": "🎬 LLM Manga Director",
}

print("### Loading: LLM Manga Director ###")
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']