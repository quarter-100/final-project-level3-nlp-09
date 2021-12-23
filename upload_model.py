from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

#HUGGINGFACE_AUTH_TOKEN = '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'

model = AutoModelForSequenceClassification.from_pretrained('rockmiin/ko-boolq-model')
model.push_to_hub(
    "BoolQ_dain_test",
    use_temp_dir=True, 
    organization="quarter100"
    )


tokenizer = AutoTokenizer.from_pretrained('rockmiin/ko-boolq-model')
tokenizer.push_to_hub(
    "BoolQ_dain_test",
    use_temp_dir=True, 
    organization="quarter100"
    )