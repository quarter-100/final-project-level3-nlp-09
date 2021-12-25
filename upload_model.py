from transformers import AutoModelForSequenceClassification, AutoTokenizer

HUGGINGFACE_AUTH_TOKEN = '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'

model = AutoModelForSequenceClassification.from_pretrained('quarter100/BoolQ_dain_test')
model.push_to_hub(
    "BoolQ_dain_test",
    use_temp_dir=True, 
    organization="quarter100",
    token = HUGGINGFACE_AUTH_TOKEN
    )


tokenizer = AutoTokenizer.from_pretrained('quarter100/BoolQ_dain_test')
tokenizer.push_to_hub(
    "BoolQ_dain_test",
    use_temp_dir=True, 
    organization="quarter100",
    token = HUGGINGFACE_AUTH_TOKEN
    )