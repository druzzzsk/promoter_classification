import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
import tqdm
import json
import os

df = pd.read_csv(r'C:\Users\User\PROJECTS\chem_ai_project\data\stong_weak_promoter.csv')

unique_sequences = df['sequence'].unique()
labels = df['label']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model.eval()

    results = []

    for sequence in tqdm.tqdm(unique_sequences):
        tokens = tokenizer(sequence, return_tensors='pt', padding=True, truncation=True).to(device)

        if tokens['input_ids'].shape[1] > tokenizer.model_max_length:
            print(f"Skipping long sequence of length {len(sequence)}")
            continue

        outputs = model(**tokens)
        hidden_states = outputs[0].squeeze(0)

        emb = torch.mean(hidden_states, dim=0)
        emb = emb / emb.norm()  

        results.append({
            "input": sequence,
            "embedding": emb.tolist()
        })

json_path = "gene_hyenadna.json"
with open(json_path, 'w') as f:
    json.dump(results, f)

with open(json_path, "r") as f:
    data = json.load(f)

df = pd.json_normalize(data) 

embedding_df = pd.DataFrame(df["embedding"].tolist())
embedding_df.columns = [f"emb_{i}" for i in range(embedding_df.shape[1])]

final_df = pd.concat([df["input"], embedding_df], axis=1)
final_df = final_df.rename(columns={"input": "sequence"})
final_df['label'] = labels

final_csv_path = "strong_weak_promoter_bert_embeds.csv"
final_df.to_csv(final_csv_path, index=False)

print(f"Файл сохранён: {os.path.abspath(final_csv_path)}")