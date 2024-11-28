import time
import torch
from pipe_utils import AverageMeter, ProgressMeter
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def run_model_test(dataloader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    top = AverageMeter('Pearson', ':1.3f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, top])

    model.eval()
    preds, actuals = [], []

    with torch.no_grad():
        end = time.time()
        for idx, batch in enumerate(dataloader):
            bsz = batch[0].size(0)
            batch = tuple(t.cuda() for t in batch) if torch.cuda.is_available() else batch

            # Ajustado para entradas do RoBERTa
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1]
            }

            # Obter logits diretamente
            outputs = model(**inputs)
            logits = outputs.logits.view(-1)  # Ajustado para saída de logits
            predictions = logits
            true_labels = batch[2].view(-1)  # Alterado para batch[2], correspondendo aos labels

            # Armazenar métricas para comparação
            preds.extend(predictions.cpu().detach().numpy().tolist())
            actuals.extend(true_labels.cpu().detach().numpy().tolist())

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

    return top.avg, preds, actuals
