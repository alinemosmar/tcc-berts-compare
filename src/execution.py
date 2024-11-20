import pandas as pd
from sklearn.model_selection import train_test_split
#divisão dos dados, somente executados 1x

urls = [
    "https://raw.githubusercontent.com/sidleal/porsimplessent/master/pss/pss2_align_length_ori_nat.tsv",
    "https://raw.githubusercontent.com/sidleal/porsimplessent/master/pss/pss2_align_length_nat_str.tsv",
    "https://raw.githubusercontent.com/sidleal/porsimplessent/master/pss/pss2_align_length_ori_str.tsv"
]

# Mapear URLs para os respectivos níveis de simplicidade
simplicity_map = {
    urls[0]: 2,  # ori_nat
    urls[1]: 1,  # nat_str
    urls[2]: 3   # ori_str
}

# Carregar datasets, adicionar a coluna simplicity_level e concatená-los
dfs = []
for url in urls:
    df = pd.read_csv(url, sep='\t', on_bad_lines='skip')
    df['simplicity_level'] = simplicity_map[url]
    dfs.append(df)

porsimples_df = pd.concat(dfs, ignore_index=True)

# Ajustar a simplicidade para 0 onde 'changed' é 'N'
porsimples_df.loc[porsimples_df['changed'] == 'N', 'simplicity_level'] = 0

# Dividir o dataset em 80% treino, 10% validação e 10% teste, mantendo proporções iguais dos três arquivos
train_df, temp_df = train_test_split(porsimples_df, test_size=0.2, stratify=porsimples_df['simplicity_level'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['simplicity_level'], random_state=42)

# Selecionar apenas as colunas 'sentence_text_from', 'sentence_text_to' e 'simplicity_level'
train_df = train_df[['sentence_text_from', 'sentence_text_to', 'simplicity_level']]
val_df = val_df[['sentence_text_from', 'sentence_text_to', 'simplicity_level']]
test_df = test_df[['sentence_text_from', 'sentence_text_to', 'simplicity_level']]

# Calcular a porcentagem de cada nível de simplicidade
def calculate_percentage(df, name):
    percentage = df['simplicity_level'].value_counts(normalize=True) * 100
    print(f"Porcentagem de cada Simplicity Level no {name}:")
    print(percentage)
    print("\n")

calculate_percentage(train_df, "Conjunto de Treinamento")
calculate_percentage(val_df, "Conjunto de Validação")
calculate_percentage(test_df, "Conjunto de Teste")

# Salvar os datasets processados
train_df.to_csv('train.csv', index=False)
val_df.to_csv('val.csv', index=False)
test_df.to_csv('test.csv', index=False)
