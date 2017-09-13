#!/usr/bin/env python3
import math

import numpy as np
import torch
from torchtext import data
from torchtext import datasets
import argparse

CUDA_AVAILABLE = torch.cuda.is_available()

EMBEDDING_DIM = 32
HIDDEN_SIZE = 64
NUM_LAYER = 2
DROPOUT = 0.1
BIDIRECTIONAL = True
DEFAULT_BATCH_SIZE = 32


# num_embeddings, embedding_dim, padding_idx, input_size, hidden_size, num_layers, dropout, bidirectional):

def main():
  parser = argparse.ArgumentParser(description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--embedding-size', type=int, dest="embedding_size", help="Embedding size", default=EMBEDDING_DIM)
  parser.add_argument('--hidden-size', type=int, dest="hidden_size", help="Hidden size", default=HIDDEN_SIZE)
  parser.add_argument('--nlayers', type=int, dest="nlayers", help="Number of RNN layers", default=NUM_LAYER)
  parser.add_argument('-b', '--batch-size', type=int, dest="batch_size", help="Batch size", default=DEFAULT_BATCH_SIZE)
  parser.add_argument('--learning-rate', type=float, dest="learning_rate", help="Initial learning rate", default=0.1)
  parser.add_argument('--learning-rate-decay', type=float, dest="learning_rate_decay", help="Learning rate decay",
    default=0.5)
  parser.add_argument('--epochs', type=int, default=5,
    help="Start decaying every epoch after and including this epoch.")
  parser.add_argument('--start-decay-at', dest="start_decay_at", type=int, default=3,
    help="Start decaying every epoch after and including this epoch.")
  parser.add_argument('--batches-per-print', type=int, dest="batches_per_print", help="Number of batches per print",
    default=100)
  parser.add_argument('-m', '--model', help="Path to the model file to load", default=None)
  parser.add_argument('--data', help="train or test", default="train")
  cmd_args = parser.parse_args()

  src = data.Field(include_lengths=True, tokenize=list)
  tgt = data.Field(include_lengths=True, tokenize=list)

  mt_train = datasets.TranslationDataset(path='data/%s' % cmd_args.data, exts=('.src', '.tgt'), fields=(src, tgt))
  mt_dev = datasets.TranslationDataset(path='data/dev', exts=('.src', '.tgt'), fields=(src, tgt))

  print("Building vocabularies..")
  src.build_vocab(mt_train)
  tgt.build_vocab(mt_train)

  print("Making batches..")
  # sort key 부호는 GPU에서는 -, CPU에서는 +를 붙여야 하나?
  SIGN = -1 if CUDA_AVAILABLE else 1
  train_iter = data.BucketIterator(
    dataset=mt_train, batch_size=cmd_args.batch_size,
    device=(None if CUDA_AVAILABLE else -1),
    repeat=False,
    sort_key=lambda x: len(x.src) * SIGN
  )
  dev_iter = data.BucketIterator(
    dataset=mt_dev, batch_size=cmd_args.batch_size,
    device=(None if CUDA_AVAILABLE else -1),
    repeat=False, train=False,
    sort_key=lambda x: len(x.src) * SIGN
  )

  print("Creating model..")
  from spacer import Spacer
  num_classes = len(tgt.vocab)
  padding_idx = tgt.vocab.stoi["<pad>"]
  model = Spacer(len(src.vocab), num_classes, cmd_args.embedding_size, cmd_args.hidden_size, cmd_args.nlayers, DROPOUT,
    BIDIRECTIONAL, padding_idx=padding_idx)
  if CUDA_AVAILABLE:
    model.cuda(0)

  if cmd_args.model:
    print("Loading model: {}".format(cmd_args.model))
    state_dict = torch.load(cmd_args.model)
    model.load_state_dict(state_dict)

  criterion = torch.nn.CrossEntropyLoss(ignore_index=padding_idx)
  learning_rate = cmd_args.learning_rate

  loss_history = []
  for epoch in range(1, cmd_args.epochs + 1):
    if epoch >= cmd_args.start_decay_at and len(loss_history) > 1 and loss_history[-1] > loss_history[-2]:
      learning_rate *= cmd_args.learning_rate_decay
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_losses = []
    correct_answer_count = 0
    total_question_count = 0
    for batch_idx, batch in enumerate(train_iter):
      optimizer.zero_grad()

      inputs, src_length = batch.src
      y_ = model(inputs, src_length)
      y_ = y_.view(-1, num_classes)
      y = batch.trg[0]
      y = y.view(-1)
      loss = criterion(y_, y)
      loss.backward()
      optimizer.step()

      train_losses.append(loss.data[0])
      _, prediction = torch.max(y_, dim=1)
      total_question_count += prediction.size()[0]
      correct_answer_count += (prediction == y).float().sum().data[0]
      if batch_idx % cmd_args.batches_per_print == 0:
        average_loss = np.mean(train_losses)

        print(
          "{}-{}(BS: {}), TrainLoss: {:.5f}, Accuracy: {:.5f}, LR:{:.5f}".format(epoch, batch_idx, cmd_args.batch_size,
            average_loss, correct_answer_count / total_question_count, learning_rate))
        print("Sentence: {}".format("".join(src.vocab.itos[x[0]] for x in batch.src[0].data)))
        prediction = prediction.view(-1, batch.batch_size)
        print("Prediction: {}".format("".join(tgt.vocab.itos[x[0]] for x in prediction.data)))
        y = y.view(-1, batch.batch_size)
        print("Answer    : {}".format("".join(tgt.vocab.itos[x[0]] for x in y.data)))

        train_losses = []
        correct_answer_count = 0
        total_question_count = 0

    cv_losses = []
    for cv_batch in dev_iter:
      inputs, src_length = cv_batch.src
      y_ = model(inputs, src_length)
      y_ = y_.view(-1, num_classes)
      y = cv_batch.trg[0]
      y = y.view(-1)
      loss = criterion(y_, y)
      cv_losses.append(loss.data[0])
    cv_average_loss = np.mean(cv_losses)
    loss_history.append(cv_average_loss)

    filename = "models/spacer_{:02d}_{:.4f}.pth".format(epoch, cv_average_loss)
    print("Saving a file: {}".format(filename))
    torch.save(model.state_dict(), filename)

    print("== Summary ==")
    for i, l in enumerate(loss_history, start=1):
      print("Epoch: {}, CV Loss: {}".format(i, l))
    print("")

  print('done')


if __name__ == "__main__":
  main()
