import sys
import numpy as np
import torch
import time
import warnings
import torch.backends.cudnn as cudnn
from utils import get_args, get_dataloaders, save_checkpoint
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from torch.optim import AdamW
from torch.nn import MSELoss
from sklearn.preprocessing import StandardScaler

from transformers import AutoTokenizer
from sub_reward_model import SubRewardModel
from subreward_training_pipeline import train_subreward, validate_subreward

warnings.filterwarnings("ignore")
best_val_metric = None

def main_worker(gpu, args):
    global best_val_metric
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    model = SubRewardModel()

    tokenizer = AutoTokenizer.from_pretrained('roberta-base', do_lower_case=False)

    if not torch.cuda.is_available():
        print('Using CPU, this will be slow')
    else:
        model = torch.nn.DataParallel(model).cuda()

    criterion = MSELoss().cuda()
    optimizer = AdamW(model.parameters(),
                      lr=args.learning_rate,
                      weight_decay=args.weight_decay)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=-1)

    dataloader = get_dataloaders(args.data_folder, tokenizer, args.batch_size, args.workers,
                                 args.max_seq_length)

    training_loader = dataloader['loader']['training']
    validation_loader = dataloader['loader']['validation']

    for epoch in range(args.start_epoch, args.epochs):
        time1 = time.time()
        train_loss, train_metric, preds, actuals = train_subreward(training_loader, model, optimizer, criterion, scheduler,
                                                                   epoch, args)

        time2 = time.time()
        print('Training epoch {}, total time {:.2f}, loss {:.7f}'.format(epoch, (time2 - time1), train_loss))

        val_time1 = time.time()
        val_loss, val_metric, val_preds, val_actuals = validate_subreward(validation_loader, model, criterion, epoch, args)
        val_time2 = time.time()
        print('Validation epoch {}, total time {:.2f}, loss {:.7f}'.format(epoch, (val_time2 - val_time1), val_loss))

        if best_val_metric is None or val_metric > best_val_metric:
            print('Updating checkpoint. New best found.')
            best_val_metric = val_metric
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_val_metric': best_val_metric,
                'optimizer': optimizer.state_dict(),
            }, args.save_folder, 'checkpoint_{}.pth.tar'.format(args.estimator))


if __name__ == '__main__':
    __args = get_args()

    if __args.seed is not None:
        torch.manual_seed(__args.seed)
        torch.cuda.manual_seed_all(__args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('Você escolheu definir uma semente para o treinamento. Isso ativará o modo determinístico do CUDNN, '
                      'o que pode reduzir o desempenho do treinamento. Você pode ver um comportamento inesperado ao reiniciar '
                      'de checkpoints.')

    if __args.gpu is not None:
        warnings.warn('Você escolheu uma GPU específica. Isso desabilitará completamente o data parallelism.')

    main_worker(__args.gpu, __args)

    print("Done!")
    sys.exit(0)
