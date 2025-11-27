"""
Model Handler for Local LLM Inference
Supports CPU and GPU execution with Hugging Face Transformers
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class ModelHandler:
    """Handles local LLM inference for code remediation"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct", use_gpu: bool = False):
        """
        Initialize the model handler
        
        Args:
            model_name: HuggingFace model identifier
            use_gpu: Whether to use GPU if available
        """
        self.model_name = model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        
        logger.info(f"Initializing model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            logger.info("Trying alternative model: deepseek-ai/deepseek-coder-1.3b-instruct")
            model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
            self.model_name = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate settings
        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "use_safetensors": True  # Force safetensors to avoid torch.load vulnerability
        }
        
        if self.use_gpu:
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = torch.float16
        else:
            # For CPU, use float32
            model_kwargs["torch_dtype"] = torch.float32
        
        logger.info("Loading model... This may take a few minutes.")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Memory error detected. Trying much smaller model: Salesforce/codegen-350M-mono")
            try:
                model_name = "Salesforce/codegen-350M-mono"
                self.model_name = model_name
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float32,
                    use_safetensors=True
                )
            except Exception as e2:
                logger.error(f"Failed to load codegen-350M: {e2}")
                logger.info("Trying minimal model: bigcode/tiny_starcoder_py")
                model_name = "bigcode/tiny_starcoder_py"
                self.model_name = model_name
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float32,
                    use_safetensors=True
                )
        
        if not self.use_gpu:
            self.model = self.model.to(self.device)
        
        self.model.eval()
        logger.info("Model loaded successfully!")
    
    def generate_fix(self, prompt: str, max_new_tokens: int = 128, 
                     temperature: float = 0.1, top_p: float = 0.95) -> Dict:
        """
        Generate code fix using the local LLM
        
        Args:
            prompt: Input prompt with vulnerability details
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Dict containing generated text and token counts
        """
        try:
            # Tokenize input with shorter context for speed
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024  # Reduced from 2048 for faster processing
            ).to(self.device)
            
            input_token_count = inputs['input_ids'].shape[1]
            
            # Generate response with optimized settings for speed
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=False,  # Greedy decoding is faster
                    num_beams=1,  # No beam search for speed
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True  # Enable KV cache for faster generation
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            output_token_count = outputs.shape[1] - input_token_count
            
            return {
                'text': generated_text,
                'input_tokens': input_token_count,
                'output_tokens': output_token_count
            }
            
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'gpu_available': torch.cuda.is_available(),
            'using_gpu': self.use_gpu
        }


class ModelHandlerPipeline:
    """Alternative implementation using HuggingFace pipeline (simpler but less control)"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct", use_gpu: bool = False):
        """Initialize using pipeline approach"""
        self.model_name = model_name
        self.device = 0 if use_gpu and torch.cuda.is_available() else -1
        
        logger.info(f"Initializing pipeline for: {model_name}")
        
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            device=self.device,
            dtype=torch.float16 if self.device >= 0 else torch.float32,
            trust_remote_code=True
        )
        
        logger.info("Pipeline initialized successfully!")
    
    def generate_fix(self, prompt: str, max_new_tokens: int = 1024) -> Dict:
        """Generate using pipeline"""
        result = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            return_full_text=False
        )
        
        generated_text = result[0]['generated_text']
        
        # Approximate token counts (pipeline doesn't expose them directly)
        input_tokens = len(prompt.split())  # Rough estimate
        output_tokens = len(generated_text.split())  # Rough estimate
        
        return {
            'text': generated_text,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens
        }