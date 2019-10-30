#-*- coding: utf-8 -*-
import queue
import torch
import random
import os
import torch.nn as nn
import torch.optim as optim
from funcCall import *
from loader import label_loader
from models import EncoderRNN, DecoderRNN, Seq2seq
from hyperParams import HyperParams

global target_dict
global char2index
global index2char
global SOS_token
global EOS_token
global PAD_token
DATASET_PATH = './data/'
target_dict = dict()

hyper_p = HyperParams()
hyper_p.input_params()

char2index, index2char = label_loader.load_label('./hackathon.labels')
SOS_token = char2index['<s>']
EOS_token = char2index['</s>']
PAD_token = char2index['_']

random.seed(hyper_p.seed)
torch.manual_seed(hyper_p.seed)
torch.cuda.manual_seed_all(hyper_p.seed)
cuda = not hyper_p.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

feature_size = 40  # MFCC n_mfcc = 40이라 40

enc = EncoderRNN(feature_size, hyper_p.hidden_size ,
                 input_dropout_p = hyper_p.dropout, dropout_p = hyper_p.dropout,
                 n_layers = hyper_p.layer_size,
                 bidirectional = hyper_p.bidirectional, rnn_cell = 'gru', variable_lengths = False)

dec = DecoderRNN(len(char2index), hyper_p.max_len, hyper_p.hidden_size * (2 if hyper_p.bidirectional else 1),
                 SOS_token, EOS_token,
                 n_layers = hyper_p.layer_size, rnn_cell = 'gru', bidirectional = hyper_p.bidirectional,
                 input_dropout_p = hyper_p.dropout, dropout_p = hyper_p.dropout, use_attention = hyper_p.attention)

model = Seq2seq(enc, dec)
model.flatten_parameters()
model = nn.DataParallel(model).to(device) # 병렬처리 부분인 듯

# Adam Algorithm
optimizer = optim.Adam(model.module.parameters(), lr = hyper_p.lr)
# CrossEntropy로 loss 계산
criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_token).to(device)

# 데이터 로드 start
data_list = os.path.join(DATASET_PATH, 'train_data', 'data_list.csv')
wav_paths = list()
script_paths = list()

with open(data_list, 'r') as f:
    for line in f:
        # line: "aaa.wav,aaa.label"
        wav_path, script_path = line.strip().split(',') # wav_path 여기 있음!!
        wav_paths.append(os.path.join(DATASET_PATH, 'train_data', wav_path))
        script_paths.append(os.path.join(DATASET_PATH, 'train_data', script_path))

best_loss = 1e10
best_cer = 1.0
begin_epoch = 0

# load all target scripts for reducing disk i/o
target_path = os.path.join(DATASET_PATH, 'train_label')
load_targets(target_path, target_dict)

# 데이터 로드 end
train_batch_num, train_dataset_list, valid_dataset = split_dataset(hyper_p, wav_paths,
                                                                   script_paths,
                                                                   valid_ratio = 0.05,
                                                                   target_dict = target_dict)


logger.info('start')
train_begin = time.time()


for epoch in range(begin_epoch, hyper_p.max_epochs):
    train_queue = queue.Queue(hyper_p.workers * 2)

    train_loader = MultiLoader(train_dataset_list, train_queue, hyper_p.batch_size, hyper_p.workers)
    train_loader.start()

    if epoch == 25:
        optimizer = optim.Adam(model.module.parameters(), lr = 0.00005 )

    train_loss, train_cer = train(model, train_batch_num,
                                  train_queue, criterion,
                                  optimizer, device,
                                  train_begin, hyper_p.workers,
                                  10, hyper_p.teacher_forcing)

    logger.info('Epoch %d (Training) Loss %0.4f CER %0.4f' % (epoch, train_loss, train_cer))

    train_loader.join()

    valid_queue = queue.Queue(hyper_p.workers * 2)
    valid_loader = BaseDataLoader(valid_dataset, valid_queue, hyper_p.batch_size, 0)
    valid_loader.start()

    eval_loss, eval_cer = evaluate(model, valid_loader, valid_queue, criterion, device)
    logger.info('Epoch %d (Evaluate) Loss %0.4f CER %0.4f' % (epoch, eval_loss, eval_cer))

    valid_loader.join()

    best_loss_model = (eval_loss < best_loss)
    best_cer_model = (eval_cer < best_cer)

    if best_loss_model:
        torch.save(model, "./best_loss")

    if best_cer_model:
        torch.save(model, "./best_cer")