from transformers import T5ForConditionalGeneration, T5Tokenizer
import gdown
import zipfile
import os
import config

def download_and_extract(file_id, file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=True)
        temp_zip_path = os.path.join(file_path, "temp.zip")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, temp_zip_path, quiet=False)
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            for file in zip_ref.namelist():
                zip_ref.extract(file, file_path)
                extracted_file_path = os.path.join(file_path, file)
                if os.path.isdir(extracted_file_path):
                    continue  
                os.rename(extracted_file_path, os.path.join(file_path, os.path.basename(file)))
        os.remove(temp_zip_path)

def load_model():
    download_and_extract(config.MODEL_ID, config.MODEL_PATH)
    download_and_extract(config.TOKENIZER_ID, config.TOKENIZER_PATH)
    tokenizer = T5Tokenizer.from_pretrained(config.TOKENIZER_PATH)
    model = T5ForConditionalGeneration.from_pretrained(config.MODEL_PATH)
    return tokenizer, model