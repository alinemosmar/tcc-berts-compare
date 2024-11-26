from transformers import AutoConfig, AutoModel
import torch.nn as nn

class SubRewardModel(nn.Module):

    def __init__(self, dropout=0.2):
        super(SubRewardModel, self).__init__()
        config = AutoConfig.from_pretrained('neuralmind/bert-base-portuguese-cased')
        config.hidden_dropout_prob = dropout
        config.attention_probs_dropout_prob = dropout
        self.bert = AutoModel.from_pretrained('neuralmind/bert-base-portuguese-cased', config=config)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(config.hidden_size, 1)  # Utilize config.hidden_size
        )
        self.loss_fct = nn.MSELoss()

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output  # Utilize o pooler_output para representação [CLS]
        pooled_output = self.dropout(pooled_output)
        logits = self.regressor(pooled_output).clamp(-1, 1)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1), labels.view(-1))
            return loss, logits
        else:
            return logits