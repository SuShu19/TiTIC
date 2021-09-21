import torch
import torch.nn as nn
from RQ1 import periodic_activations

# build LSTM
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm=nn.LSTM(input_size=300,           # input_size=word'featureVectors=300
                          hidden_size=256,          # hidden_size=256
                          num_layers=1,             # num of running layer
                          batch_first=True)         # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        self.fc=nn.Linear(256,2)                    #full conntion

    def forward(self,x):
        # x shape (batch, time_step, input_size)
        # out shape (batch, time_step, output_size)
        # hidden shape (n_layers, batch, hidden_size)
        # cell shape (n_layers, batch, hidden_size)
        out,(hidden,cell)=self.lstm(x)
        out=self.fc(out[:,-1,:])                    # use the last time_step to input full conntion
        out=torch.sigmoid(out)                      # make output in (0,1)
        return out

# build LSTM
class single_LSTM(nn.Module):
    def __init__(self):
        super(single_LSTM, self).__init__()
        self.lstm=nn.LSTM(input_size=1,           # input_size=word'featureVectors=300
                          hidden_size=256,          # hidden_size=256
                          num_layers=1,             # num of running layer
                          batch_first=True)         # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        self.fc=nn.Linear(256,3)                    #full conntion

    def forward(self,x):
        # x shape (batch, time_step, input_size)
        # out shape (batch, time_step, output_size)
        # hidden shape (n_layers, batch, hidden_size)
        # cell shape (n_layers, batch, hidden_size)
        out,(hidden,cell)=self.lstm(x)
        out=self.fc(out[:,-1,:])                    # use the last time_step to input full conntion
        out=torch.sigmoid(out)                      # make output in (0,1)
        return out

    def generate_vec(self, x):
        return self.lstm(x)

# build LSTM
class w2v_t2v_LSTM(nn.Module):
    def __init__(self, test_d, time_d):
        super(w2v_t2v_LSTM, self).__init__()
        self.time2vec = periodic_activations.SineActivation(1, time_d)
        self.lstm=nn.LSTM(input_size=test_d+time_d,           # input_size=word'featureVectors=300
                          hidden_size=256,          # hidden_size=256
                          num_layers=1,             # num of running layer
                          batch_first=True)         # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        self.fc=nn.Linear(256,3)                    #full conntion
        self.sm = nn.Softmax(dim=1)

    def forward(self,x_time, x_txt):
        x_ti = self.time2vec(x_time)
        x = torch.cat([x_ti, x_txt], dim=2)
        out,(hidden,cell)=self.lstm(x)
        out=self.fc(out[:,-1,:])                    # use the last time_step to input full conntion
        # out=torch.sigmoid(out)                      # make output in (0,1)
        out = self.sm(out)
        return out

    def vector(self,x_time, x_txt):
        x_ti = self.time2vec(x_time)
        x = torch.cat([x_ti, x_txt], dim=2)
        return x
# build LSTM
class w2v_t2v_LSTM_binary(nn.Module):
    def __init__(self):
        super(w2v_t2v_LSTM_binary, self).__init__()
        self.time2vec = periodic_activations.SineActivation(1, 40)
        self.lstm=nn.LSTM(input_size=300+40,           # input_size=word'featureVectors=300
                          hidden_size=256,          # hidden_size=256
                          num_layers=1,             # num of running layer
                          batch_first=True)         # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        self.fc=nn.Linear(256,2)                    #full conntion
        self.sm = nn.Softmax(dim=1)

    def forward(self,x_time, x_txt):
        x_ti = self.time2vec(x_time)
        x = torch.cat([x_ti, x_txt], dim=2)
        out,(hidden,cell)=self.lstm(x)
        out=self.fc(out[:,-1,:])                    # use the last time_step to input full conntion
        out = self.sm(out)
        return out

    def vector(self,x_time, x_txt):
        x_ti = self.time2vec(x_time)
        x = torch.cat([x_ti, x_txt], dim=2)
        return x

# build RNN
class w2v_t2v_RNN(nn.Module):
    def __init__(self):
        super(w2v_t2v_RNN, self).__init__()
        self.time2vec = periodic_activations.SineActivation(1, 40)
        self.rnn=nn.RNN(input_size=300+40,           # input_size=word'featureVectors=300
                          hidden_size=256,          # hidden_size=256
                          num_layers=1,             # num of running layer
                          batch_first=True)         # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        self.fc=nn.Linear(256,3)                    #full conntion

    def forward(self,x_time, x_txt):
        # x shape (batch, time_step, input_size)
        # out shape (batch, time_step, output_size)
        # hidden shape (n_layers, batch, hidden_size)
        # cell shape (n_layers, batch, hidden_size)
        x_ti = self.time2vec(x_time)
        x = torch.cat([x_ti, x_txt], dim=2)
        out,(hidden,cell)=self.rnn(x)
        out=self.fc(out[:,-1,:])                    # use the last time_step to input full conntion
        out=torch.sigmoid(out)                      # make output in (0,1)
        assert torch.isnan(out).sum() == 0, print(out, x_time)
        return out


# build LSTM
class w2v_LSTM(nn.Module):
    def __init__(self, text_d):
        super(w2v_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=text_d,  # input_size=word'featureVectors=300
                            hidden_size=256,  # hidden_size=256
                            num_layers=1,  # num of running layer
                            batch_first=True)  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        self.fc = nn.Linear(256, 3)  # full conntion
        self.sm = nn.Softmax(dim=1)


    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # out shape (batch, time_step, output_size)
        # hidden shape (n_layers, batch, hidden_size)
        # cell shape (n_layers, batch, hidden_size)
        out, (hidden, cell) = self.lstm(x)
        out = self.fc(out[:, -1, :])  # use the last time_step to input full conntion
        out = self.sm(out)
        return out





# build LSTM
class t2v_LSTM(nn.Module):
    def __init__(self, time_dim):
        super(t2v_LSTM, self).__init__()
        self.t2v = periodic_activations.SineActivation(1, time_dim)
        self.lstm = nn.LSTM(input_size=time_dim, hidden_size=time_dim, num_layers=1, batch_first=True)
        self.fc=nn.Linear(time_dim,3)                    #full conntion
        self.sm = nn.Softmax(dim=1)
        # self.fc=nn.Linear(50,2)                    #full conntion

    def forward(self, x):
        out = self.t2v(x)
        out, (hidden, cell) = self.lstm(out)
        out = self.fc(out[:,-1,:])
        # out = torch.sigmoid(out)
        out = self.sm(out)
        return out

    def generate_vec(self, x):
        return self.t2v(x)