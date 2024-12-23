import time
import torch
from pipe_utils import AverageMeter, ProgressMeter
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
#pipeline de teste ajustado para o bertimbau

def run_model_test(dataloader, model, args, tokenizer):
    batch_time = AverageMeter('Time', ':6.3f')
    top = AverageMeter('Pearson', ':1.3f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, top])

    model.eval()
    preds, actuals = [], []
    sentence_from, sentence_to = [], []

    with torch.no_grad():
        end = time.time()
        for idx, batch in enumerate(dataloader):
            bsz = batch[0].size(0)
            batch = tuple(t.cuda() for t in batch) if torch.cuda.is_available() else batch

            # Atualização para incluir token_type_ids e labels
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],  # Inclusão de token_type_ids
                'labels': batch[3]           # Inclusão de labels
            }

            output = model(**inputs)
            # Supondo que output seja uma tupla (loss, logits)
            logits = output[1].view(-1)
            predictions = output[1].view(-1)
            true_labels = batch[3].view(-1)  # Atualizado para batch[3]

            # Armazenar métricas para comparação
            preds.extend(predictions.cpu().detach().numpy().tolist())
            actuals.extend(true_labels.cpu().detach().numpy().tolist())
            decoded_sentences = tokenizer.batch_decode(batch[0], skip_special_tokens=True)
            for sentence in decoded_sentences:
                from_to = sentence.split('[SEP]')
                sentence_from.append(from_to[0].strip())
                sentence_to.append(from_to[1].strip() if len(from_to) > 1 else '')        

            corr, _ = pearsonr(preds, actuals)

            top.update(corr, bsz)
            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % args.print_freq == 0:
                progress.display(idx)

    # Cálculo das métricas ao final do teste
    mse = mean_squared_error(actuals, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, preds)

    print(f'Test Results: MSE={mse:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}, Pearson Correlation={top.avg:.4f}')

    return top.avg, preds, actuals,sentence_from, sentence_to
