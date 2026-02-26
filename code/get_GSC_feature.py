import json
import os
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_poi_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def format_poi_texts(poi_texts):
    formatted_texts = []
    for poi in poi_texts:
        formatted_text = poi.replace('“', '').replace('”', '').replace(':', ' ').replace('–', ' ')
        formatted_texts.append(formatted_text)
    return formatted_texts


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)


def get_bert_embeddings(poi_texts):
    if not poi_texts:
        return None

    inputs = tokenizer(poi_texts, return_tensors='pt', padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    cls_embeddings = outputs.last_hidden_state[:, 0, :]
    return cls_embeddings


def save_embeddings(cls_embeddings, save_path):
    if cls_embeddings is not None:
        cls_embeddings_np = cls_embeddings.cpu().numpy()
        np.save(save_path, cls_embeddings_np)



file_path = 'metadata.json'

poi_data = load_poi_data(file_path)

save_folder = 'outputs/poi_features'
os.makedirs(save_folder, exist_ok=True)

for data in poi_data:
    poi_texts = data["poi_texts"]
    formatted_poi_texts = format_poi_texts(poi_texts)
    cls_embeddings = get_bert_embeddings(formatted_poi_texts)

    if cls_embeddings is not None:
        save_path = os.path.join(save_folder, f"{data['id']}_poi_cls_embeddings.npy")
        save_embeddings(cls_embeddings, save_path)

print("Done")

