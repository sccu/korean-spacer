import torch
from torch.autograd import Variable

if torch.cuda.is_available():
  import torch.cuda as tc
else:
  import torch as tc


class Spacer(torch.nn.Module):
  def __init__(self, num_embeddings, num_out, embedding_dim, hidden_size, num_layers, dropout,
      bidirectional, padding_idx=-100):
    print("Initializing super()...")
    super(Spacer, self).__init__()
    self.num_layers = num_layers
    self.num_directions = 2 if bidirectional else 1
    self.hidden_size = hidden_size
    self.padding_idx = padding_idx
    print("Creating Embedding...")
    self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
    print("Creating LSTM...")
    self.lstm = torch.nn.LSTM(embedding_dim, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
    print("Creating Linear...")
    self.linear = torch.nn.Linear(hidden_size * self.num_directions, num_out)
    # print("Creating LogSoftmax...")
    # self.softmax = torch.nn.LogSoftmax()

    # num_layers * num_directions, batch, hidden_size

  def init_hiddens(self, batch_size):
    h0 = Variable(tc.FloatTensor(self.num_layers * self.num_directions, batch_size, self.hidden_size).zero_(),
      requires_grad=False)
    c0 = Variable(tc.FloatTensor(self.num_layers * self.num_directions, batch_size, self.hidden_size).zero_(),
      requires_grad=False)
    return h0, c0

  def forward(self, input, src_length):
    embedded = self.embedding(input)
    # src_length = input.data.ne(self.padding_idx).long().sum(0, keepdim=False).squeeze().tolist()
    packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, src_length.tolist())

    batch_size = input.size(1)
    h0, c0 = self.init_hiddens(batch_size)
    out, _ = self.lstm(packed, (h0, c0))
    out, _ = torch.nn.utils.rnn.pad_packed_sequence(out)
    out = self.linear(out)
    # size = out.size()
    # out = out.view(-1, out.size()[2])
    # out = self.softmax(out)
    # out = out.view(size)
    return out
