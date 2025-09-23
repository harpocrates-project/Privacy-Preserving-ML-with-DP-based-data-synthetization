import torch.nn as nn


class Encoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder, self).__init__()

    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim, self.hidden_dim2 = embedding_dim, 6 * embedding_dim, 2 * embedding_dim

    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    
    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=self.hidden_dim2,
      num_layers=1,
      batch_first=True
    )

    self.rnn3 = nn.LSTM(
      input_size= self.hidden_dim2,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )

  def forward(self, x):
    batch_size = x.shape[0]

    x = x.reshape((batch_size, self.seq_len, self.n_features))

    x, (_, _) = self.rnn1(x)

    x, (hidden_n, _) = self.rnn2(x)

    x, (hidden_n, _) = self.rnn3(x)

    return hidden_n.reshape((batch_size, self.embedding_dim))

class Decoder(nn.Module):

  def __init__(self, seq_len, input_dim=64, n_features=1):
    super(Decoder, self).__init__()

    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.hidden_dim2, self.n_features = 2 * input_dim, 6 * input_dim,  n_features

    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True
    )

    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )

    self.rnn3 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=self.hidden_dim2,
      num_layers=1,
      batch_first=True
    )

    self.output_layer = nn.Linear(self.hidden_dim2, n_features)

  def forward(self, x):
    batch_size = x.shape[0]
  
    x = x.unsqueeze(1).repeat(1, self.seq_len, 1) 
 
    x, (hidden_n, cell_n) = self.rnn1(x)
   
    x, (hidden_n, cell_n) = self.rnn2(x)

    x, (hidden_n, cell_n) = self.rnn3(x)

    return self.output_layer(x)



class RecurrentAutoencoder(nn.Module):

  def __init__(self, seq_len, n_features, device, embedding_dim=64):
    super(RecurrentAutoencoder, self).__init__()

    self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
    self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x