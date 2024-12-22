import os
import numpy as np
import pandas as pd
import torch
import time
import warnings
from sklearn.preprocessing import StandardScaler
from utils import get_args, get_dataloaders, save_checkpoint
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.nn import MSELoss
from sub_reward_model import SubRewardModel
from subreward_training_pipeline import train_subreward, validate_subreward

warnings.filterwarnings("ignore")
best_val_metric = None

def main_worker(gpu, args):
    global best_val_metric
    args.gpu = gpu

    current_dir = os.path.dirname(os.path.abspath(__file__))

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    model = SubRewardModel()
    tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False)

    if not torch.cuda.is_available():
        print('Using CPU, this will be slow')
    else:
        model = torch.nn.DataParallel(model).cuda()

    criterion = MSELoss().cuda()
    optimizer = AdamW(model.parameters(),
                      lr=args.learning_rate,
                      weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=-1)

    dataloader = get_dataloaders(args.data_folder, tokenizer, args.batch_size, args.workers, args.max_seq_length)
    training_loader = dataloader['loader']['training']
    validation_loader = dataloader['loader']['validation']

    # Ajustar o scaler no conjunto de treinamento
    scaler = StandardScaler()
    all_actuals = []
    for batch in training_loader:
        actuals = batch[3].numpy()  # Extraindo valores reais
        all_actuals.extend(actuals)
    scaler.fit(np.array(all_actuals).reshape(-1, 1))

    for epoch in range(args.start_epoch, args.epochs):
        time1 = time.time()
        train_loss, train_metric, preds, actuals, sentence_from, sentence_to = train_subreward(
            training_loader, model, optimizer, criterion, scheduler, epoch, args, tokenizer
        )
        time2 = time.time()
        print(f'Training epoch {epoch}, total time {time2 - time1:.2f}, loss {train_loss:.7f}')

        # Inversão de escala para previsões e valores reais
        no_scaled_actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()
        no_scaled_preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

        # Salvar previsões e valores reais no CSV
        df = pd.DataFrame({
            'sentence_from': sentence_from,
            'sentence_to': sentence_to,
            'predicted_simplicity': no_scaled_preds,
            'actual_simplicity': no_scaled_actuals
        })
        df.to_csv(os.path.join(current_dir, f'val_predictions_epoch_{epoch}.csv'), index=False)

        val_time1 = time.time()
        val_loss, val_metric, val_preds, val_actuals = validate_subreward(
            validation_loader, model, criterion, epoch, args
        )
        val_time2 = time.time()
        print(f'Validation epoch {epoch}, total time {val_time2 - val_time1:.2f}, loss {val_loss:.7f}')

        # Atualizar checkpoint se encontrar melhor métrica de validação
        if best_val_metric is None or val_metric > best_val_metric:
            print('Updating checkpoint. New best found.')
            best_val_metric = val_metric
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_val_metric': best_val_metric,
                'optimizer': optimizer.state_dict(),
            }, args.save_folder, f'checkpoint_{args.estimator}.pth.tar')


if __name__ == '__main__':
    __args = get_args()

    if __args.seed is not None:
        torch.manual_seed(__args.seed)
        torch.cuda.manual_seed_all(__args.seed)
        warnings.warn(
            'Você escolheu definir uma semente para o treinamento. Isso ativará o modo determinístico do CUDNN, '
            'o que pode reduzir o desempenho do treinamento.'
        )

    if __args.gpu is not None:
        warnings.warn('Você escolheu uma GPU específica. Isso desabilitará completamente o data parallelism.')

    main_worker(__args.gpu, __args)
    print("Done!")
