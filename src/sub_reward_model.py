from transformers import AutoConfig, AutoModel
import torch.nn as nn

class SubRewardModel(nn.Module):
    def __init__(self, dropout=0.2):
        super(SubRewardModel, self).__init__()
        # Configuração do JurisBERT
        config = AutoConfig.from_pretrained('alfaneo/jurisbert-base-portuguese-sts')
        config.hidden_dropout_prob = dropout
        config.attention_probs_dropout_prob = dropout

        # Carregar o modelo JurisBERT
        self.bert = AutoModel.from_pretrained('alfaneo/jurisbert-base-portuguese-sts', config=config)

        # Camada de dropout e regressão
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(config.hidden_size, 1)  # Previsão de simplicidade como saída única
        )
        self.loss_fct = nn.MSELoss()  # Função de perda para regressão

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        # Forward no modelo JurisBERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # Usar embeddings médios para obter representação da sentença
        pooled_output = outputs.last_hidden_state.mean(dim=1)  # Mean pooling sobre os tokens

        # Aplicar dropout e regressão
        pooled_output = self.dropout(pooled_output)
        logits = self.regressor(pooled_output).clamp(-1, 1)  # Prever valores entre -1 e 1

        if labels is not None:
            # Calcular perda se os rótulos forem fornecidos
            loss = self.loss_fct(logits.view(-1), labels.view(-1))
            return loss, logits
        else:
            # Apenas retornando as previsões para inferência
            return logits
