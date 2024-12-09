from transformers import T5ForConditionalGeneration, T5Tokenizer
import config

def load_model():
    tokenizer = T5Tokenizer.from_pretrained(config.TOKENIZER_PATH)
    model = T5ForConditionalGeneration.from_pretrained(config.MODEL_PATH)
    return tokenizer, model