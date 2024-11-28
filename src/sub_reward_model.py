from transformers import RobertaForSequenceClassification, RobertaConfig
import torch.nn as nn

class SubRewardModel(nn.Module):
    def __init__(self, dropout=0.2):
        super(SubRewardModel, self).__init__()
        # Configuração do modelo RoBERTa
        config = RobertaConfig.from_pretrained('roberta-base')
        config.num_labels = 1  # Regressão (1 saída contínua)
        config.hidden_dropout_prob = dropout
        config.attention_probs_dropout_prob = dropout

        # Carregar RoBERTa pré-treinado
        self.roberta = RobertaForSequenceClassification.from_pretrained('roberta-base', config=config)

        # Alterar a função de perda para regressão
        self.loss_fct = nn.MSELoss()

    def forward(self, input_ids, attention_mask, labels=None):
        # Forward no modelo RoBERTa
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        # Logits do modelo
        logits = outputs.logits.clamp(-1, 1)  # Previsão entre -1 e 1 (se necessário)

        if labels is not None:
            # Cálculo da perda para treinamento
            loss = self.loss_fct(logits.view(-1), labels.view(-1))
            return loss, logits
        else:
            # Retorno apenas da previsão para inferência
            return logits
