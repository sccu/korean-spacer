import torch
from torch.autograd import Variable

if torch.cuda.is_available():
  import torch.cuda as tc
else:
  import torch as tc


class Spacer(torch.nn.Module):
  def __init__(self, num_embeddings, embedding_dim, hidden_size, num_layers, dropout,
      bidirectional):
    super().__init__()
    self.num_layers = num_layers
    self.num_directions = 2 if bidirectional else 1
    self.hidden_size = hidden_size
    self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
    self.lstm = torch.nn.LSTM(embedding_dim, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
    self.linear = torch.nn.Linear(hidden_size * self.num_directions, 1)

    #num_layers * num_directions, batch, hidden_size

  def init_hiddens(self, batch_size):
    h0 = Variable(tc.zeros([self.num_layers * self.num_directions, batch_size, self.hidden_size]), requires_grad=False)
    c0 = Variable(tc.zeros([self.num_layers * self.num_directions, batch_size, self.hidden_size]), requires_grad=False)
    return h0, c0

  def forward(self, input):
    l = self.embedding(input)
    batch_size = input.size()[0]
    h0, c0 = self.init_hiddens(batch_size)
    out, (h_n, c_n) = self.lstm(l, (h0, c0))
    out = self.linear(out)
    return out
