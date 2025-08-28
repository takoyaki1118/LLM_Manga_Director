# /ComfyUI/custom_nodes/LLM_Manga_Director/nodes.py

import os
import json
import folder_paths
import re # 出力のクリーンアップ用

# --- ライブラリのインポートチェック ---
try:
    from llama_cpp import Llama
    Llama # pyflakes fix
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    # ローカルLLMが必須になったため、見つからない場合はノード全体を機能させない
    print("FATAL: llama-cpp-python is not installed. LLM_Manga_Director node will not be available.")
    print("Please install it via: pip install llama-cpp-python")
    LLAMA_CPP_AVAILABLE = False


# --- グローバル変数とヘルパー関数 ---
llm_models_dir = os.path.join(folder_paths.models_dir, "llm_models")
cache_dir = os.path.join(folder_paths.get_output_directory(), "llm_director_cache")

if not os.path.exists(llm_models_dir):
    os.makedirs(llm_models_dir)

if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

def get_llm_models():
    """/models/llm_models/ 内のGGUFファイルを探してリストを返す"""
    if not LLAMA_CPP_AVAILABLE or not os.path.isdir(llm_models_dir):
        return []
    return [f for f in os.listdir(llm_models_dir) if f.endswith((".gguf", ".GGUF"))]

# --------------------------------------------------------------------
# ノード本体: LLM Manga Director
# --------------------------------------------------------------------
class LLM_Manga_Director:
    _loaded_local_model = None
    _loaded_model_name = ""

    @classmethod
    def INPUT_TYPES(s):
        # ライブラリがない場合は、エラーメッセージを表示して何もしない
        if not LLAMA_CPP_AVAILABLE:
            return { "required": { "error": ("STRING", {"default": "llama-cpp-python not found. Please install it.", "multiline": True}) }}

        llm_models = get_llm_models()
        if not llm_models:
             llm_models = ["No models found in models/llm_models"]

        return {
            "required": {
                "job_id": ("STRING", {"default": "my_manga_project"}),
                "panel_index": ("INT", {"default": 1, "min": 1}),
                "story_brief": ("STRING", {
                    "multiline": True,
                    "default": "A brave warrior encounters a goblin in a dark forest. The warrior is initially surprised but quickly prepares for battle by drawing their sword."
                }),
                "character_sheet": ("STRING", {
                    "multiline": True,
                    "default": "1boy, warrior, silver armor, red hair, brave personality"
                }),
                "style_and_genre": ("STRING", {
                    "multiline": True,
                    "default": "manga style, black and white, dramatic shadows, dynamic camera angles, shonen battle manga"
                }),
                "local_model": (llm_models, ),
                "max_tokens": ("INT", {"default": 150, "min": 10, "max": 4096}),
                "gpu_layers": ("INT", {"default": -1, "min": -1, "max": 1000}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "story_log_for_debug")
    FUNCTION = "direct_story"
    CATEGORY = "Manga Toolbox/LLM"

    def direct_story(self, job_id, panel_index, story_brief, character_sheet, style_and_genre, local_model, max_tokens, gpu_layers):
        if not LLAMA_CPP_AVAILABLE:
            return ("ERROR: llama-cpp-python is not installed.", "Please check the console for installation instructions.")

        job_cache_dir = os.path.join(cache_dir, job_id)
        os.makedirs(job_cache_dir, exist_ok=True)
        log_file_path = os.path.join(job_cache_dir, "story_log.json")

        story_log = {}
        if panel_index == 1:
            if os.path.exists(log_file_path):
                os.remove(log_file_path)
            print(f"LLM Director: New story started for job '{job_id}'. History cleared.")
        else:
            if os.path.exists(log_file_path):
                with open(log_file_path, 'r', encoding='utf-8') as f:
                    story_log = json.load(f)
        
        previous_panels_text = "\n".join([f"- Panel {k}: {v}" for k, v in story_log.items()])
        if not previous_panels_text:
            previous_panels_text = "This is the first panel."

        # ★改善点: LLMへの指示（メタプロンプト）をより厳密に
        meta_prompt = f"""You are a professional manga director. Based on the following information, generate a prompt for Panel {panel_index} that logically follows the story's continuity.

# Overall Plot for this Scene
{story_brief}

# Character Sheet
{character_sheet}

# Overall Style and Genre
{style_and_genre}

# Story So Far
{previous_panels_text}

# Your Task
Based on all the context above, output a single line of prompt for Panel {panel_index}. The prompt should be a concise comma-separated list of English keywords (maximum 15 keywords). Stick strictly to the provided story brief, character sheet, and style—do not add unrelated elements or details not mentioned. Focus only on essential elements for this specific panel. Do not add any other explanations or text like "Prompt for Panel X:".
"""
        
        generated_prompt = self.run_local_mode(meta_prompt, local_model, max_tokens, gpu_layers)

        # ★改善点: 出力されたプロンプトを強制的にクリーンアップ
        # 冒頭の "Prompt:", "Panel X:" などの余計な接頭辞を削除
        cleaned_prompt = re.sub(r'^(prompt|panel\s*\d*)\s*:\s*', '', generated_prompt, flags=re.IGNORECASE).strip()
        # プロンプトとして不適切な可能性のある特殊文字を除去（アンダースコアはLoRA等で使うため維持）
        cleaned_prompt = re.sub(r'[^a-zA-Z0-9_, -]', '', cleaned_prompt)
        
        # 重複を除去しつつ、順序は（ある程度）維持する
        keywords = []
        seen_keywords = set()
        for kw in cleaned_prompt.split(','):
            kw_stripped = kw.strip().lower()
            if kw_stripped and kw_stripped not in seen_keywords:
                keywords.append(kw_stripped)
                seen_keywords.add(kw_stripped)
        
        # 最大15個に制限
        final_prompt = ', '.join(keywords[:15])

        # ログを更新
        story_log[str(panel_index)] = final_prompt
        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(story_log, f, indent=2, ensure_ascii=False)
            
        debug_log_text = f"--- Context for Panel {panel_index} ---\n{previous_panels_text}"

        print(f"LLM Director: Generated prompt for Panel {panel_index}: {final_prompt}")

        return (final_prompt, debug_log_text)

    def run_local_mode(self, meta_prompt, model_name, max_tokens, n_gpu_layers):
        """ローカルGGUFモデルモードでLLMを実行"""
        if "No models found" in model_name:
             return "LOCAL_ERROR: No GGU_models found in models/llm_models folder."
        
        model_path = os.path.join(llm_models_dir, model_name)
        if not os.path.exists(model_path):
            return f"LOCAL_ERROR: Model file not found at {model_path}"

        # モデルのキャッシュとロード
        if LLM_Manga_Director._loaded_model_name != model_name:
            print(f"LLM Director: Loading local model: {model_name}")
            try:
                # 以前のモデルをメモリから解放
                if LLM_Manga_Director._loaded_local_model is not None:
                    del LLM_Manga_Director._loaded_local_model
                
                LLM_Manga_Director._loaded_local_model = Llama(
                    model_path=model_path,
                    n_gpu_layers=n_gpu_layers,
                    verbose=False,
                    n_ctx=4096
                )
                LLM_Manga_Director._loaded_model_name = model_name
            except Exception as e:
                return f"LOCAL_ERROR: Failed to load model: {e}"
        
        llm = LLM_Manga_Director._loaded_local_model
        try:
            # LLMに合わせたプロンプトテンプレート（Llama3-Instruct形式）
            prompt_template = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{meta_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
            output = llm(
                prompt_template,
                max_tokens=max_tokens,
                stop=["<|eot_id|>", "<|end_of_text|>"],
                echo=False,
                temperature=0.3, # ハルシネーション（AIの暴走）を抑制
            )
            return output["choices"][0]["text"]
        except Exception as e:
            return f"LOCAL_ERROR: Failed during generation: {e}"