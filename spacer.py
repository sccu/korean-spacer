import torch

if torch.cuda.is_available():
  import torch.cuda as tc
else:
  import torch as tc


class Spacer(torch.nn.Module):
  def __init__(self, num_embeddings, embedding_dim, input_size, hidden_size, num_layers, dropout,
      bidirectional):
    super()
    self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
    self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
    self.linear = torch.nn.Linear(hidden_size, num_embeddings)

  def forward(self, input):
    pass
