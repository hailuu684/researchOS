import os
from openai import OpenAI
from researchos_backend.modal_llm import extract_json
from typing import Dict, Any, Optional
from transformers import pipeline
import torch

class BaseLLMChat:
    """Class Gốc chứa logic dùng chung cho tất cả các loại LLM Client."""
    
    def _format_strict_user(self, user: str) -> str:
        """Tiêm lệnh ép buộc xuất định dạng JSON (dùng chung cho mọi mô hình)."""
        return (
            f"{user}\n\n"
            "CRITICAL INSTRUCTION: You MUST output ONLY a valid JSON object starting with { and ending with }. "
            "Do NOT write any prose or markdown."
        )

    def complete(self, system: str, user: str, max_new_tokens: Optional[int] = None) -> str:
        """Interface bắt buộc các class con phải implement."""
        raise NotImplementedError("Class con phải tự định nghĩa hàm complete()")

    def complete_json(self, system: str, user: str, max_new_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Tự động gọi sinh text và parse thành Dictionary."""
        text = self.complete(system, user, max_new_tokens)
        return extract_json(text)


class ModalLLM(BaseLLMChat):
    """Class trung gian chuyên xử lý giao tiếp qua OpenAI SDK API."""
    
    def __init__(self, base_url: str, api_key: str, model_name: str, temperature: float, max_tokens: int):
        self.model_name = model_name
        self.temperature = temperature
        self.default_max_tokens = max_tokens
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def complete(self, system: str, user: str, max_new_tokens: Optional[int] = None) -> str:
        max_tokens = max_new_tokens or self.default_max_tokens
        strict_user = self._format_strict_user(user)
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": strict_user},
        ]

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=int(max_tokens),
            temperature=self.temperature,
            stream=False,
        )
        
        return completion.choices[0].message.content


# =====================================================================
# CÁC CLASS CLIENT CỤ THỂ (IMPLEMENTATION)
# =====================================================================

class LocalLLMChat(ModalLLM):
    """1. Client kết nối tới Hugging Face Router."""
    def __init__(self):
        api_key = os.getenv("HF_AUTH")
        if not api_key:
            raise ValueError("❌ Không tìm thấy HF_AUTH trong file .env")
            
        model_name = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-32B-Instruct:featherless-ai")
        print(f"🔗 Đang kết nối tới HF Router: {model_name}...")
        
        super().__init__(
            base_url="https://router.huggingface.co/v1",
            api_key=api_key,
            model_name=model_name,
            temperature=0.3,
            max_tokens=1500
        )


class vLLM(ModalLLM):
    """2. Client kết nối tới vLLM / Ollama Server qua API."""
    def __init__(self):
        base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1")
        api_key = os.getenv("LOCAL_LLM_API_KEY", "EMPTY")
        model_name = os.getenv("LOCAL_LLM_MODEL_VLLM", "Qwen/Qwen2.5-32B-Instruct")
        
        print(f"🏠 Đang kết nối tới vLLM Server tại {base_url} (Model: {model_name})...")
        
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            temperature=0.2,
            max_tokens=2048
        )


class LocalServerLLMChat(BaseLLMChat):
    """3. Chạy trực tiếp Pipeline trên RAM/VRAM của máy local (Kế thừa BaseLLMChat, dùng Singleton)."""
    
    _instance = None
    
    def __new__(cls):
        # Singleton pattern: Đảm bảo model 32B chỉ được load 1 lần duy nhất
        if cls._instance is None:
            cls._instance = super(LocalServerLLMChat, cls).__new__(cls)
            cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        self.model_name = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-32B-Instruct")
        self.default_max_tokens = 2048
        
        print(f"⏳ Đang tải mô hình {self.model_name} trực tiếp vào GPU. Quá trình này sẽ mất vài phút...")
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model_name,
            model_kwargs={"dtype": torch.bfloat16}, 
            device_map="auto"
        )
        print("✅ Đã load Transformers Pipeline thành công!")

    def complete(self, system: str, user: str, max_new_tokens: Optional[int] = None) -> str:
        max_tokens_int = int(max_new_tokens or self.default_max_tokens)
        local_llm_temperature = float(os.getenv("LOCAL_LLM_TEMPERATURE", 0.2))
        # Dùng lại hàm format json prompt từ class Gốc
        strict_user = self._format_strict_user(user)
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": strict_user},
        ]

        print("🤖 Transformers Pipeline đang suy luận...")
        outputs = self.pipe(
            messages,
            max_new_tokens=max_tokens_int,
            temperature=local_llm_temperature,
            do_sample=True,
            return_full_text=False # Bỏ qua phần prompt trong output
        )
        
        return outputs[0]["generated_text"].strip()