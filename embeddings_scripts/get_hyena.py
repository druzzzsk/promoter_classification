import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
import tqdm
import json
from transformers import AutoConfig
from typing import Optional
import torch.nn as nn
from huggingface_hub import snapshot_download
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch

df = pd.read_csv(r'C:\Users\User\PROJECTS\chem_ai_project\data\promoter_or_non.csv')
labels = df['label'].tolist()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

access_token = "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"  # Replace with your actual token

with torch.no_grad():
    # Load HyenaDNA model
    model = AutoModelForCausalLM.from_pretrained("LongSafari/hyenadna-large-1m-seqlen-hf", trust_remote_code=True, token=access_token).to(device)
    tokenizer = AutoTokenizer.from_pretrained("LongSafari/hyenadna-large-1m-seqlen-hf", trust_remote_code=True, token=access_token)  # Use same tokenizer as model
    model.eval() # Set model to evaluation mode


    results = []
    unique_sequences = df['sequence'].unique()

    for sequence in tqdm.tqdm(unique_sequences):
        # Tokenize the sequence
        enc = tokenizer(sequence, return_tensors="pt", truncation=True, padding=True).to(device)

        # Get embeddings from the last hidden state
        outputs = model(**enc, output_hidden_states=True)  # Pass tokenized input through model
        hidden_states = outputs.hidden_states[-1]  # Access the last hidden state

        emb = torch.mean(hidden_states.cpu()[0], dim=0)  # Average over sequence length


        results.append({
            "input": sequence,
            "embedding": emb.tolist()
        })

json_path = "hyenadna.json"
with open('hyenadna.json', "w") as file:
    json.dump(list(results), file)

# 5. Загрузка из JSON и преобразование в DataFrame
with open(json_path, "r") as f:
    data = json.load(f)

df = pd.json_normalize(data)  # Получим колонки 'input' и 'embedding'

# 6. Разделение embedding на отдельные колонки
embedding_df = pd.DataFrame(df["embedding"].tolist())
embedding_df.columns = [f"emb_{i}" for i in range(embedding_df.shape[1])]

# 7. Объединение с исходной последовательностью
final_df = pd.concat([df["input"], embedding_df], axis=1)
final_df = final_df.rename(columns={"input": "sequence"})
final_df['label'] = labels

# 8. Сохранение в CSV
final_csv_path = "promoter_or_non_hyena_embeds.csv"
final_df.to_csv(final_csv_path, index=False)

print(f"Файл сохранён: {os.path.abspath(final_csv_path)}")    
