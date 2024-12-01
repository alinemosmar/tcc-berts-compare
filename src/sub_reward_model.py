from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch.nn as nn
import torch

class SubRewardModel(nn.Module):
    def __init__(self, dropout=0.2):
        super(SubRewardModel, self).__init__()
        # Configuração do modelo Bertimbau-Law
        config = AutoConfig.from_pretrained('juridics/bertimbaulaw-base-portuguese-sts-scale')
        config.hidden_dropout_prob = dropout
        config.attention_probs_dropout_prob = dropout

        # Carregar o modelo base Bertimbau-Law
        self.bert = AutoModel.from_pretrained('juridics/bertimbaulaw-base-portuguese-sts-scale', config=config)
        self.dropout = nn.Dropout(dropout)

        # Camada de regressão linear
        self.regressor = nn.Sequential(
            nn.Linear(config.hidden_size, 1)  # Saída de um único valor para regressão
        )

        # Função de perda para tarefas de regressão
        self.loss_fct = nn.MSELoss()

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        # Passar os dados pelo modelo Bertimbau-Law
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids  # Incluído token_type_ids
        )

        # Obter a representação média (Mean Pooling)
        token_embeddings = outputs.last_hidden_state  # Todas as embeddings dos tokens
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled_output = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        pooled_output = self.dropout(pooled_output)

        # Regressor linear para prever a simplicidade
        logits = self.regressor(pooled_output).clamp(-1, 1)

        # Cálculo da perda, se os rótulos forem fornecidos
        if labels is not None:
            loss = self.loss_fct(logits.view(-1), labels.view(-1))
            return loss, logits
        else:
            return logits
