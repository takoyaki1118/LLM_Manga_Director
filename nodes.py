# /ComfyUI/custom_nodes/LLM_Manga_Director/nodes.py

import os
import json
import folder_paths

# --- ライブラリのインポートチェック ---
# ユーザーが必要なライブラリをインストールしているか確認し、なければエラーメッセージを出す
try:
    from llama_cpp import Llama
    Llama # pyflakes fix
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    print("Warning: llama-cpp-python is not installed. Local LLM mode will not be available.")
    print("Please install it via: pip install llama-cpp-python")
    LLAMA_CPP_AVAILABLE = False

try:
    import requests
    requests # pyflakes fix
    REQUESTS_AVAILABLE = True
except ImportError:
    print("Warning: requests library is not installed. API mode will not be available.")
    print("Please install it via: pip install requests")
    REQUESTS_AVAILABLE = False


# --- グローバル変数とヘルパー関数 ---

# LLMモデル(GGUF)を置くためのディレクトリパス
llm_models_dir = os.path.join(folder_paths.models_dir, "llm_models")

# キャッシュファイルを保存するためのディレクトリパス
cache_dir = os.path.join(folder_paths.get_output_directory(), "llm_director_cache")

# llm_modelsディレクトリが存在しない場合は作成
if not os.path.exists(llm_models_dir):
    os.makedirs(llm_models_dir)

# キャッシュディレクトリが存在しない場合は作成
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

def get_llm_models():
    """/models/llm_models/ 内のGGUFファイルを探してリストを返す"""
    if not os.path.isdir(llm_models_dir):
        return []
    return [f for f in os.listdir(llm_models_dir) if f.endswith((".gguf", ".GGUF"))]

# --------------------------------------------------------------------
# ノード本体: LLM Manga Director
# --------------------------------------------------------------------
class LLM_Manga_Director:
    """
    LLMを使用して、漫画のストーリーに基づいたプロンプトをコマごとに生成するノード。
    APIモードとローカルGGUFモデルモードをサポート。
    """
    _loaded_local_model = None
    _loaded_model_name = ""

    @classmethod
    def INPUT_TYPES(s):
        # 利用可能なGGUFモデルのリストを取得
        llm_models = get_llm_models()
        
        # 必須の入力項目
        required_inputs = {
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
             "execution_mode": (["API (OpenAI compatible)", "Local (GGUF Model)"],),
        }

        # オプショナル（モードによって表示/非表示）な入力項目
        optional_inputs = {
            # APIモード用
            "api_endpoint": ("STRING", {"default": "https://api.openai.com/v1/chat/completions"}),
            "api_model_name": ("STRING", {"default": "gpt-4-turbo"}),
            "api_key": ("STRING", {"default": "YOUR_API_KEY_HERE"}),
            # ローカルモード用
            "local_model": (llm_models, ),
            "max_tokens": ("INT", {"default": 200, "min": 10, "max": 4096}),
            "gpu_layers": ("INT", {"default": -1, "min": -1, "max": 1000}),
        }

        # ライブラリの利用可否に応じて選択肢を調整
        if not REQUESTS_AVAILABLE:
            required_inputs["execution_mode"] = (["Local (GGUF Model)"],)
        if not LLAMA_CPP_AVAILABLE:
            required_inputs["execution_mode"] = (["API (OpenAI compatible)"],)
        if not REQUESTS_AVAILABLE and not LLAMA_CPP_AVAILABLE:
             raise Exception("Neither 'requests' nor 'llama-cpp-python' is available. This node cannot function.")


        return {"required": required_inputs, "optional": optional_inputs}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "story_log_for_debug")
    FUNCTION = "direct_story"
    CATEGORY = "Manga Toolbox/LLM"

    def direct_story(self, job_id, panel_index, story_brief, character_sheet, style_and_genre, execution_mode, **kwargs):
        job_cache_dir = os.path.join(cache_dir, job_id)
        os.makedirs(job_cache_dir, exist_ok=True)
        log_file_path = os.path.join(job_cache_dir, "story_log.json")

        # --- ログ（文脈）の管理 ---
        story_log = {}
        if panel_index == 1:
            # 最初のコマなら、古いログを削除してリセット
            if os.path.exists(log_file_path):
                os.remove(log_file_path)
            print(f"LLM Director: New story started for job '{job_id}'. History cleared.")
        else:
            # 2コマ目以降なら、ログを読み込む
            if os.path.exists(log_file_path):
                with open(log_file_path, 'r', encoding='utf-8') as f:
                    story_log = json.load(f)
        
        # --- LLMへの指示（メタプロンプト）の組み立て ---
        previous_panels_text = "\n".join([f"- Panel {k}: {v}" for k, v in story_log.items()])
        if not previous_panels_text:
            previous_panels_text = "This is the first panel."

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
Based on all the context above, output a single line of prompt for Panel {panel_index}. The prompt should be a comma-separated list of English keywords. Do not add any other explanations or text.
"""
        
        # --- モードに応じてLLMを実行 ---
        generated_prompt = ""
        if execution_mode == "API (OpenAI compatible)":
            if not REQUESTS_AVAILABLE:
                raise Exception("'requests' library is not installed. API mode is unavailable.")
            generated_prompt = self.run_api_mode(meta_prompt, **kwargs)

        elif execution_mode == "Local (GGUF Model)":
            if not LLAMA_CPP_AVAILABLE:
                raise Exception("'llama-cpp-python' is not installed. Local mode is unavailable.")
            generated_prompt = self.run_local_mode(meta_prompt, **kwargs)

        # --- 結果の処理と保存 ---
        # LLMの出力から余計な部分（例："panel 2 prompt: "など）を取り除く
        final_prompt = generated_prompt.strip().replace("\"", "")

        # ログを更新
        story_log[str(panel_index)] = final_prompt
        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(story_log, f, indent=2, ensure_ascii=False)
            
        debug_log_text = f"--- Context for Panel {panel_index} ---\n{previous_panels_text}"

        print(f"LLM Director: Generated prompt for Panel {panel_index}: {final_prompt}")

        return (final_prompt, debug_log_text)

    def run_api_mode(self, meta_prompt, **kwargs):
        """APIモードでLLMを実行"""
        endpoint = kwargs.get("api_endpoint")
        model = kwargs.get("api_model_name")
        api_key = kwargs.get("api_key")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": meta_prompt}],
            "temperature": 0.7,
        }

        try:
            response = requests.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error calling API: {e}")
            return f"API_ERROR: {e}"

    def run_local_mode(self, meta_prompt, **kwargs):
        """ローカルGGUFモデルモードでLLMを実行"""
        model_name = kwargs.get("local_model")
        max_tokens = kwargs.get("max_tokens")
        n_gpu_layers = kwargs.get("gpu_layers")
        
        if not model_name:
             return "LOCAL_ERROR: No GGUF model selected."

        model_path = os.path.join(llm_models_dir, model_name)
        if not os.path.exists(model_path):
            return f"LOCAL_ERROR: Model file not found at {model_path}"

        # --- モデルのキャッシュとロード ---
        # 既にロードされているモデルと設定が同じであれば、再ロードしない
        if LLM_Manga_Director._loaded_model_name != model_name:
            print(f"Loading local model: {model_name}")
            try:
                LLM_Manga_Director._loaded_local_model = Llama(
                    model_path=model_path,
                    n_gpu_layers=n_gpu_layers,
                    verbose=False,
                    n_ctx=4096 # コンテキストウィンドウのサイズ
                )
                LLM_Manga_Director._loaded_model_name = model_name
            except Exception as e:
                return f"LOCAL_ERROR: Failed to load model: {e}"
        
        # --- プロンプト生成 ---
        llm = LLM_Manga_Director._loaded_local_model
        try:
            # LLMに合わせたプロンプトテンプレートを使用（Llama3-Instruct形式）
            prompt_template = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{meta_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
            output = llm(
                prompt_template,
                max_tokens=max_tokens,
                stop=["<|eot_id|>", "<|end_of_text|>"],
                echo=False,
            )
            return output["choices"][0]["text"]
        except Exception as e:
            return f"LOCAL_ERROR: Failed during generation: {e}"