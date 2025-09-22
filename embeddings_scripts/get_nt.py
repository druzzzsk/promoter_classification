import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm  

tokenizer = AutoTokenizer.from_pretrained(
    "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", trust_remote_code=True
)
model = AutoModelForMaskedLM.from_pretrained(
    "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", trust_remote_code=True
)
model.eval() 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Используем устройство:", device)

def compute_mean_embeddings(sequences, max_length=82):
 
    inputs = tokenizer.batch_encode_plus(
        sequences,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
    
    last_hidden_states = outputs.hidden_states[-1]  

    attention_mask_expanded = attention_mask.unsqueeze(-1) 
    summed = torch.sum(last_hidden_states * attention_mask_expanded, dim=1)
    counts = torch.sum(attention_mask_expanded, dim=1)
    mean_embeddings = summed / counts  

    return mean_embeddings.cpu().numpy() 

def embed_sequences_df(df, batch_size=4, max_length=82):
    sequences = df["sequence"].tolist()
    labels = df["label"].tolist()
    
    all_embeddings = []

    for i in tqdm(range(0, len(sequences), batch_size), desc="Embedding batches"):
        print(f"Обрабатывается батч {i // batch_size + 1}")
        batch_seqs = sequences[i:i+batch_size]
        batch_embeddings = compute_mean_embeddings(batch_seqs, max_length=max_length)
        all_embeddings.append(batch_embeddings)
    
    all_embeddings = np.vstack(all_embeddings)  

    embedding_dim = all_embeddings.shape[1]
    embedding_columns = [f"emb_{i}" for i in range(embedding_dim)]

    result_df = pd.concat([
        df.reset_index(drop=True)[["sequence", "label"]],
        pd.DataFrame(all_embeddings, columns=embedding_columns)
    ], axis=1)

    return result_df

if __name__ == "__main__":

    input_df = pd.read_csv(r'C:\Users\User\PROJECTS\chem_ai_project\data\promoter_or_non.csv')

    # Получаем эмбеддинги
    output_df = embed_sequences_df(input_df)

    print("\n Итоговый DataFrame:")
    print(output_df.head())
    output_df.to_csv("strong_or_weak_promoter_nt_embeddings.csv", index=False)
