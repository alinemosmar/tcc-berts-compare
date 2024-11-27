from transformers import AutoConfig, AutoModel
import torch.nn as nn

class SubRewardModel(nn.Module):

    def __init__(self, dropout=0.2):
        super(SubRewardModel, self).__init__()
        # Configuração do modelo BERTimbau
        config = AutoConfig.from_pretrained('felipemaiapolo/legalnlp-bert')
        config.hidden_dropout_prob = dropout
        config.attention_probs_dropout_prob = dropout

        # Carregar o modelo base BERTimbau
        self.bert = AutoModel.from_pretrained('felipemaiapolo/legalnlp-bert', config=config)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(config.hidden_size, 1)
        )
        self.loss_fct = nn.MSELoss()

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.regressor(pooled_output).clamp(-1, 1)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1), labels.view(-1))
            return loss, logits
        else:
            return logits
