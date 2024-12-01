import os
import pandas as pd
import torch
import torch.nn.utils.rnn
import transformers
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer  # Atualizado para compatibilidade com bertimbaulaw

# Tokenização e carregamento de dados
def get_simplification_data(batch_size, data_folder, max_seq_length, num_workers, tokenizer):
    # Carregar os dados do seu próprio dataset
    train_df = get_simplification_dataframe(data_folder, "train.csv")
    valid_df = get_simplification_dataframe(data_folder, "val.csv")
    test_df = get_simplification_dataframe(data_folder, "test.csv")

    train_dataset = get_simplification_dataset_from_dataframe(train_df, max_seq_length, tokenizer)
    valid_dataset = get_simplification_dataset_from_dataframe(valid_df, max_seq_length, tokenizer)
    test_dataset = get_simplification_dataset_from_dataframe(test_df, max_seq_length, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, valid_loader, test_loader

def get_simplification_dataset_from_dataframe(dataframe, max_seq_length, tokenizer):
    # Remover linhas com valores ausentes
    dataframe = dataframe.dropna(subset=['sentence_text_from', 'sentence_text_to', 'simplicity_level'])

    if max_seq_length is None:
        max_seq_length = tokenizer.model_max_length

    # Tokenização dos pares de sentenças
    inputs = tokenizer.batch_encode_plus(
        [(row['sentence_text_from'], row['sentence_text_to']) for _, row in dataframe.iterrows()],
        add_special_tokens=True,
        return_token_type_ids=True,  # Necessário para o bertimbaulaw
        return_attention_mask=True,
        max_length=max_seq_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    sent_ids = [sid for sid, _ in dataframe.iterrows()]  

    # Escalonamento dos labels de simplicidade
    scaler = StandardScaler()
    scaler.fit(dataframe['simplicity_level'].to_numpy().reshape(-1, 1))
    train_labels = scaler.transform(dataframe['simplicity_level'].to_numpy().reshape(-1, 1))
    ids = torch.tensor([sid for sid in sent_ids], dtype=torch.long)
    y = torch.tensor([label[0] for label in train_labels.tolist()], dtype=torch.float32)

    # Inclusão dos token_type_ids
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'], y, ids)

    return dataset

def get_simplification_dataframe(data_folder, filename):
    filepath = os.path.join(data_folder, filename)
    df = pd.read_csv(filepath)

    return df

