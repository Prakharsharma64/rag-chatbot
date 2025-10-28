"""
LLM handler for DeepSeek-V3.2-Exp
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import config
try:
    import streamlit as st
except ImportError:
    # Streamlit not available (for non-streamlit usage)
    st = None


class LLMHandler:
    """Handles LLM loading and text generation"""
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loaded = False
    
    def load_model(self):
        """Load the DeepSeek model"""
        if self.loaded:
            return self.tokenizer, self.model
        
        try:
            print(f"Loading model {config.LLM_MODEL_NAME}...")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.LLM_MODEL_NAME,
                trust_remote_code=True,
                cache_dir=config.CACHE_DIR
            )
            
            # Load model with optimized settings
            self.model = AutoModelForCausalLM.from_pretrained(
                config.LLM_MODEL_NAME,
                trust_remote_code=True,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                cache_dir=config.CACHE_DIR,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self.loaded = True
            
            print(f"Model loaded successfully on {self.device}")
            return self.tokenizer, self.model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    
    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95
    ) -> str:
        """
        Generate a response from the model
        """
        if not self.loaded:
            self.tokenizer, self.model = self.load_model()
        
        try:
            # Format the prompt
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # Tokenize
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs)
                )
            
            # Decode
            response = self.tokenizer.decode(
                outputs[0][inputs.shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            return f"Error during generation: {str(e)}"
