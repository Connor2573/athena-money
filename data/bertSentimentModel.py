import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased").to(device)

def getSentiment(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits

    return logits[0][0], logits[0][1]
