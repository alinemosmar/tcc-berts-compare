import os
import numpy as np
import torch
import time
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from pipe_utils import AverageMeter, ProgressMeter, get_lr
from sklearn.preprocessing import StandardScaler


def train_subreward(dataloader, model, optimizer, criterion, scheduler, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':1.5f')
    learning = AverageMeter('Learning Rate', ':1.7f')
    top = AverageMeter('Pearson Correlation', ':1.3f')
    
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, learning, losses, top],
        prefix="Epoch: [{}]".format(epoch),
        logfile=os.path.join(args.save_folder, 'log_training_' + args.model_name + '.csv'))

    preds, actuals = [], []
    l1_criterion = torch.nn.L1Loss()
    model.train()
    end = time.time()

    for idx, batch in enumerate(dataloader):
        data_time.update(time.time() - end)
        bsz = batch[0].size(0)
        batch = tuple(t.cuda() for t in batch) if torch.cuda.is_available() else batch

        # Inputs ajustados para RoBERTa
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'labels': batch[2]
        }

        outputs = model(**inputs)

        predicted_labels = outputs.logits.view(-1)
        true_labels = batch[2].view(-1)

        # Armazenar métricas para comparação
        preds.extend(predicted_labels.cpu().detach().numpy())
        actuals.extend(true_labels.cpu().detach().numpy())

        loss = criterion(predicted_labels, true_labels)

        # L1 regularization
        if args.l1_regularization > 0:
            for param in model.parameters():
                loss += args.l1_regularization * l1_criterion(param, torch.zeros_like(param))

        # Gradient accumulation
        if args.gradient_accumulation_steps > 1:
            loss /= args.gradient_accumulation_steps

        losses.update(loss.item(), bsz)
        loss.backward()

        # Atualização dos gradientes
        optimizer.step()
        scheduler.step((epoch + idx) / len(dataloader))
        optimizer.zero_grad()

        # Atualização de métricas
        mse = mean_squared_error(actuals, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(actuals, preds)

        print(f'MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}')
        corr, _ = pearsonr(preds, actuals)
        top.update(corr, bsz)
        learning.update(get_lr(optimizer), bsz)

        batch_time.update(time.time() - end)
        end = time.time()

        progress.log_metrics(idx)
        if (idx + 1) % args.print_freq == 0:
            progress.display(idx)

    return losses.avg, top.avg, preds, actuals


def validate_subreward(dataloader, model, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':1.5f')
    top = AverageMeter('Pearson Correlation', ':1.3f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, losses, top],
        prefix="Epoch: [{}]".format(epoch),
        logfile=os.path.join(args.save_folder, 'log_validation_' + args.model_name + '.csv'))

    preds, actuals = [], []
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, batch in enumerate(dataloader):
            bsz = batch[0].size(0)
            batch = tuple(t.cuda() for t in batch) if torch.cuda.is_available() else batch

            # Inputs ajustados para RoBERTa
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1]
            }

            outputs = model(**inputs)

            predicted_labels = outputs.logits.view(-1)
            true_labels = batch[2].view(-1)

            # Armazenar métricas para comparação
            preds.extend(predicted_labels.cpu().numpy().tolist())
            actuals.extend(true_labels.cpu().numpy().tolist())

            loss = criterion(predicted_labels, true_labels)
            losses.update(loss.item(), bsz)

            corr, _ = pearsonr(preds, actuals)
            top.update(corr, bsz)

            batch_time.update(time.time() - end)
            end = time.time()

            progress.log_metrics(idx)
            if (idx + 1) % args.print_freq == 0:
                progress.display(idx)

    # Cálculo das métricas ao final da validação
    mse = mean_squared_error(actuals, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, preds)

    print(f'Validation Epoch {epoch}: MSE={mse:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}')

    return losses.avg, top.avg, preds, actuals
