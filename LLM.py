import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any, Dict, Optional, Union
import os
import gc

class LanguageModelManager:
    """
    Hugging FaceのTransformersライブラリを使用して、自然言語生成（NLG）を簡単に行うためのラッパークラス。
    
    前提条件:
        - PyTorchがインストールされていること。
        - Hugging FaceのTransformersライブラリがインストールされていること。
    
    属性:
        tokenizer: Hugging FaceのAutoTokenizerインスタンス。
        model: Hugging FaceのAutoModelForCausalLMインスタンス。
    
    参考文献:
        - Tokenizerの詳細: https://huggingface.co/docs/transformers/main_classes/tokenizer
        - Modelの詳細: https://huggingface.co/docs/transformers/v4.33.0/en/main_classes/model
        
    
    使用例:
        >>> my_model = LanguageModelManager(model_name_or_path="rinna/japanese-gpt-1b")
        >>> my_generation_options = {"min_length":10,"max_length":50, "do_sample": True, "top_k": 500,  "top_p":0.95}
        >>> my_model.generate_text("おはようございます", generation_options=my_generation_options, only_answer=True)
    """
    
    def __init__(self, 
                 model_name_or_path: Union[str, os.PathLike], 
                 tokenizer_options: Optional[Dict[str, Any]] = None, 
                 model_options: Optional[Dict[str, Any]] = None) -> None:
        """
        コンストラクタで指定されたモデル名を使用して、tokenizerとmodelを初期化します。
        
        引数:
            model_name (str): 使用するモデルの名前。
            tokenizer_option (Optional[Dict[str, Any]]): Tokenizerの設定オプション。
            model_option (Optional[Dict[str, Any]]): モデルの設定オプション。
        
        戻り値:
            なし
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **(tokenizer_options or {}))
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **(model_options or {}))
        self.model_name_or_path = model_name_or_path
        
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

    def get_model_info(self) -> Dict[str, Any]:
        """
        現在のモデルに関する基本的な情報を取得します。
        
        このメソッドは、モデルの名前と総パラメータ数を含む辞書を返します。
        
        戻り値:
            Dict[str, Any]: モデルに関する情報を含む辞書。
                - "model_name_or_path": モデルの名前またはパス（str）
                - "n_params": モデルの総パラメータ数（int）
        
        """
        n_params = sum(p.numel() for p in self.model.parameters())
        return {"model_name_or_path": self.model_name_or_path, "n_params": n_params}


    def generate_text(self, 
                      text: str, 
                      encoding_options: Optional[Dict[str, Any]] = None, 
                      generation_options: Optional[Dict[str, Any]] = None, 
                      decoding_options: Optional[Dict[str, Any]] = None,
                      only_answer: bool = False) -> str:
        """
        テキストを入力として受け取り、モデルによって生成されたテキストを返します。
        
        処理のステップ:
            1. テキストをトークンにエンコード。
            2. トークンを使用してテキストを生成。
            3. 生成されたトークンをデコードしてテキストに変換。
        
        引数:
            text (str): 入力テキスト。
            encode_option (Optional[Dict[str, Any]]): エンコードオプション。
            generate_option (Optional[Dict[str, Any]]): テキスト生成オプション。
            decode_option (Optional[Dict[str, Any]]): デコードオプション。
            only_answer (bool): Trueの場合、生成されたテキストのうち、入力テキスト以降の部分のみを返します。
        
        戻り値:
            str: 生成されたテキスト。
        """
        
        default_encoding_options = {"add_special_tokens":False,"return_tensors":"pt"}
        encoding_options = encoding_options or  default_encoding_options
        
        token_ids = self.tokenizer.encode(text, **encoding_options)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=token_ids.to(self.model.device),
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                bad_words_ids=[[self.tokenizer.unk_token_id]],
                **(generation_options or {})
            )
        
        output = self.tokenizer.decode(output_ids.tolist()[0], **(decoding_options or {}))
        output = output.replace("</s>", "")
        
        if only_answer:
            return output[len(text):]
        
        return output

    def release_model_and_tokenizer_memory(self):
        """
        モデルとトークナイザーのインスタンスを削除し、メモリを解放します。
        
        このメソッドは、不要になったモデルとトークナイザーをメモリから解放する際に使用します。
        PythonのガベージコレクションとPyTorchのCUDAキャッシュもクリアされます。
        """
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()