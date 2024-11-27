import sys
import os

from utils import get_args, get_dataloaders
from sub_reward_model import SubRewardModel
from transformers import BertTokenizer
from subreward_testing_pipeline import run_model_test as test_bertg
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pipe_utils import AverageMeter, ProgressMeter, get_lr
import torch
import numpy as np
import torch.backends.cudnn as cudnn
#teste

def main_worker(args):
    # Inicializar o modelo
    model = SubRewardModel()

    tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False)


    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        model = torch.nn.DataParallel(model).cuda()

    # Verificar se há um checkpoint compatível para carregar
    checkpoint_path = os.path.join(args.save_folder, 'checkpoint_{}.pth.tar'.format(args.estimator))
    if os.path.isfile(checkpoint_path):
        model_ckpt = torch.load(checkpoint_path, map_location=args.device)
        model.load_state_dict(model_ckpt["state_dict"])
        print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, model_ckpt['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_path))

    cudnn.benchmark = True
    # get dataloaders
    dataloader = get_dataloaders(args.data_folder, tokenizer, args.batch_size, args.workers,
                                 args.max_seq_length)

    testing_loader = dataloader['loader']['testing']
    corr, preds, actuals = test_bertg(testing_loader, model, args)

    print("Correlation: {}".format(corr))
    mse = mean_squared_error(actuals, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, preds)
    

    print("Test Metrics:")
    print(f"Pearson Correlation: {corr:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

if __name__ == '__main__':
    __args = get_args()

    main_worker(__args)

    print("Done!")
    sys.exit(0)
