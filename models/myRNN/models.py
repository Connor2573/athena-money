import torch
from torch.autograd import Variable 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MV_LSTM(torch.nn.Module):
    def __init__(self,n_features,seq_length):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 128 # number of hidden states
        self.n_layers = 5 # number of LSTM layers (stacked)
    
        self.l_lstm = torch.nn.LSTM(input_size = n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers)
        # according to pytorch docs LSTM output is 
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = torch.nn.Linear(self.n_hidden, self.seq_len, 1 * self.n_hidden)
        
    
    def init_hidden(self, batches):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers,batches,self.n_hidden, dtype=torch.float32).to(device)
        cell_state = torch.zeros(self.n_layers,batches,self.n_hidden, dtype=torch.float32).to(device)
        self.hidden = (hidden_state, cell_state)
    
    
    def forward(self, x):        
        _, batch_size, seq_len = x.size()
        lstm_out, self.hidden = self.l_lstm(x,self.hidden)
        # lstm_out(with batch_first = True) is 
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest       
        # .contiguous() -> solves tensor compatibility error
        x = lstm_out.contiguous().view(batch_size,-1)
        return self.l_linear(x)


class athenaLTSM(torch.nn.Module):
    def __init__(self, num_targets, input_size, hidden_size, num_layers, seq_length):
        super(athenaLTSM, self).__init__()
        self.num_classes = num_targets #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  torch.nn.Linear(num_layers*hidden_size, 128) #fully connected 1
        self.fc = torch.nn.Linear(128, num_targets) #fully connected last layer

        self.relu = torch.nn.ReLU()

    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out