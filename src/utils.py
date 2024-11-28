import os
import argparse

import torch.cuda

from data.simplification import get_simplification_data


def get_args():
    parser = argparse.ArgumentParser(description='Arguments for training and evaluating models')

    # Required parameters
    parser.add_argument("--model_name", default='roberta-base', type=str,
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--data_folder", default='./datasets', type=str,
                        help="Path to data folder.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer than this "
                             "will be truncated, sequences shorter will be padded.")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size for training and evaluation.")
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.1, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--warmup_steps", default=1000, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--l1_regularization", default=0.0, type=float,
                        help="L1 regularization coefficient.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--print_freq", default=10, type=int,
                        help="Print frequency.")

    # lstm
    parser.add_argument("--input_size", default=128, type=int,
                        help="Size of the input vocabulary.")
    parser.add_argument("--output_size", default=128, type=int,
                        help="Size of the output vocabulary.")
    parser.add_argument("--embedding_size", default=300, type=int,
                        help="Size of the word embeddings.")
    parser.add_argument("--hidden_size", default=256, type=int,
                        help="Size of the hidden layer.")
    parser.add_argument("--num_layers", default=2, type=int,
                        help="Number of layers in the LSTM.")
    parser.add_argument("--dropout", default=0.2, type=float,
                        help="Dropout probability.")
    parser.add_argument("--pad_idx", default=0, type=int,
                        help="Index of the padding token.")


    # distributed training parameters
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed value (default: None)')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--workers', default=1, type=int,
                        help='Number of data loading workers (default: 1)')

    parser.add_argument("--estimator", choices=['grammar', 'meaning', 'simplicity', 'bart', 'seq2seq'],
                        default='grammar', type=str,
                        help="The estimator to train and evaluate.")

    args = parser.parse_args()
    args.model_path = "./save/{}_models".format(args.estimator)
    args.model_name = '{}_len_{}_bsz_{}_lr_{}'.format(args.model_name,
                                                      args.max_seq_length,
                                                      args.batch_size,
                                                      args.learning_rate)

    args.start_epoch = 0
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.save_folder = os.path.join(args.model_path, args.model_name)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    return args


def save_checkpoint(state, save_folder, filename: str):
    filename = os.path.join(save_folder, filename)
    torch.save(state, filename)


def get_dataloaders(data_folder, tokenizer, batch_size, num_workers, max_seq_length):

    train_loader, dev_loader, test_loader = None, None, None
    train_loader, dev_loader, test_loader = get_simplification_data(batch_size, data_folder, max_seq_length,
                                                                         num_workers, tokenizer)

    return {'loader': {'training': train_loader, 'validation': dev_loader, 'testing': test_loader}}
