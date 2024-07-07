import logging
import sys
from .model_util import get_model_loader, ModelLoader
from profiling import profile

TASK_NAME = 'text-generation'
MODEL_ID = 'meta-llama/Meta-Llama-3-8B-Instruct'

logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'quantize=true':
        logger.info("Using quantized model...")
        model_loader: ModelLoader = get_model_loader(MODEL_ID, quantize=True, model_directory='.models')
    else:
        logger.info("Using native (non-quantized) model")
        model_loader: ModelLoader = get_model_loader(MODEL_ID)
    
    model, tokenizer = profile(model_loader.load_model)
    
    if next(model.parameters()).is_cuda:
        logger.info("Model loaded by pipeline is running on CUDA")
    else:
        logger.info("Model loaded by pipeline is running on CPU")
    
    messages = [
        {"role": "system",
         "content": 'You are an ominous AI assistant known as "PAL."'},
        {"role": "user", "content": "Please explain why the chicken crossed the road."}
    ]
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    outputs = profile(model.generate,
                      input_ids,
                      max_new_tokens=256,
                      eos_token_id=terminators,
                      do_sample=True,
                      temperature=0.6,
                      top_p=0.9,
                      )
    
    tokenized_response = outputs[0][input_ids.shape[-1]:]
    assistant_response = tokenizer.decode(tokenized_response, skip_special_tokens=True)
    
    print(assistant_response)


if __name__ == '__main__':
    main()
