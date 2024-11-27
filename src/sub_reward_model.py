from transformers import AutoConfig, AutoModel
import torch.nn as nn

class SubRewardModel(nn.Module):
    def __init__(self, dropout=0.2):
        super(SubRewardModel, self).__init__()
        # Configuração do modelo BERTimbau
        config = AutoConfig.from_pretrained('neuralmind/bert-base-portuguese-cased')
        config.hidden_dropout_prob = dropout
        config.attention_probs_dropout_prob = dropout

        # Carregar o modelo base BERTimbau
        self.bert = AutoModel.from_pretrained('neuralmind/bert-base-portuguese-cased', config=config)
        self.dropout = nn.Dropout(dropout)

        # Camada de regressão linear
        self.regressor = nn.Sequential(
            nn.Linear(config.hidden_size, 1)  # Saída de um único valor para regressão
        )

        # Função de perda para tarefas de regressão
        self.loss_fct = nn.MSELoss()

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        # Passar os dados pelo modelo BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # Obter a representação do token [CLS]
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Pegando o token [CLS]
        pooled_output = self.dropout(pooled_output)

        # Regressor linear para prever a simplicidade
        logits = self.regressor(pooled_output).clamp(-1, 1)

        # Cálculo da perda, se os rótulos forem fornecidos
        if labels is not None:
            loss = self.loss_fct(logits.view(-1), labels.view(-1))
            return loss, logits
        else:
            return logits
