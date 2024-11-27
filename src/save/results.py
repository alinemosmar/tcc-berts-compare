import pandas as pd
import matplotlib.pyplot as plt
import os
current_dir = os.path.dirname(__file__)
bertikal_file_path = os.path.join(current_dir,"simplicity_models", "bertikal-validation.csv")
bertimbau_file_path = os.path.join(current_dir,"simplicity_models", "bertimbau-validation.csv")
#plot de métricas

df = pd.read_csv(bertimbau_file_path, sep=';', header=0)

# Converter colunas para o tipo numérico
df['epoch'] = df['epoch'].astype(int)
df['batch'] = df['batch'].astype(int)
df['time'] = df['time'].astype(float)
df['loss'] = df['loss'].astype(float)
df['pearson_correlation'] = df['pearson_correlation'].astype(float)

# Calcular a média da correlação de Pearson por época
pearson_by_epoch = df.groupby('epoch')['pearson_correlation'].mean().reset_index()

# Calcular a perda média por época
loss_by_epoch = df.groupby('epoch')['loss'].mean().reset_index()

# Encontrar a época com a melhor correlação média
best_epoch = pearson_by_epoch.loc[pearson_by_epoch['pearson_correlation'].idxmax()]
print(f"Melhor época: {int(best_epoch['epoch'])}, Correlação de Pearson média: {best_epoch['pearson_correlation']:.4f}")

# Plotar a correlação de Pearson média por época
plt.figure(figsize=(10, 6))
plt.plot(pearson_by_epoch['epoch'], pearson_by_epoch['pearson_correlation'], marker='o')
plt.title('Correlação de Pearson Média por Época')
plt.xlabel('Época')
plt.ylabel('Correlação de Pearson')
plt.grid(True)
plt.show()

# Plotar a perda média por época
plt.figure(figsize=(10, 6))
plt.plot(loss_by_epoch['epoch'], loss_by_epoch['loss'], marker='o', color='red')
plt.title('Perda Média por Época')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# Plotar correlação de Pearson e perda no mesmo gráfico
fig, ax1 = plt.subplots(figsize=(10,6))

ax1.set_xlabel('Época')
ax1.set_ylabel('Correlação de Pearson', color='blue')
ax1.plot(pearson_by_epoch['epoch'], pearson_by_epoch['pearson_correlation'], marker='o', color='blue', label='Correlação de Pearson')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Loss', color='red')
ax2.plot(loss_by_epoch['epoch'], loss_by_epoch['loss'], marker='o', color='red', label='Loss')
ax2.tick_params(axis='y', labelcolor='red')

fig.tight_layout()
plt.title('Correlação de Pearson e Loss por Época')
plt.grid(True)
plt.show()
