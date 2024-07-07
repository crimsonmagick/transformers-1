import logging
import os
import torch
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


class ModelLoader(ABC):
    
    @abstractmethod
    def load_model(self):
        pass


class NativeModelLoader(ModelLoader):
    def __init__(self, model_id):
        self.model_id = model_id
    
    def load_model(self):
        logger.info(f"Loading native model...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        return model, tokenizer


class QuantizedModelLoader(ModelLoader):
    def __init__(self, model_id, quantized_model_dir):
        self.model_id = model_id
        self.quantized_model_dir = quantized_model_dir
    
    def _quantize_model(self):
        logger.info("Quantizing model...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(self.model_id, quantization_config=quantization_config,
                                                     device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        return model, tokenizer
    
    def _save_model(self, model, tokenizer):
        logger.info(f"Saving quantized model to directory={self.quantized_model_dir}...")
        model.save_pretrained(self.quantized_model_dir)
        tokenizer.save_pretrained(self.quantized_model_dir)
        return model, tokenizer
    
    def load_model(self):
        if os.path.exists(self.quantized_model_dir):
            logger.info(f"Loading quantized model from directory={self.quantized_model_dir}...")
            model = AutoModelForCausalLM.from_pretrained(self.quantized_model_dir)
            tokenizer = AutoTokenizer.from_pretrained(self.quantized_model_dir)
        else:
            model, tokenizer = self._quantize_model()
            self._save_model(model, tokenizer)
        return model, tokenizer
