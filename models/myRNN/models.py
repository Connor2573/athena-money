import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LSTM1(torch.nn.Module):
    def __init__(self, features, hidden_layers=64):
        super(LSTM1, self).__init__()
        self.hidden_layers = hidden_layers
        # lstm1, lstm2, linear are all layers in the network
        self.lstm1 = torch.nn.LSTMCell(features, self.hidden_layers)
        self.lstm2 = torch.nn.LSTMCell(self.hidden_layers, self.hidden_layers)
        self.linear = torch.nn.Linear(self.hidden_layers, 1)
        
    def forward(self, input_t, y, future_preds=0):
        outputs, n_samples = [], y.size(0)
        h_t = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32).to(device)
        c_t = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32).to(device)
        h_t2 = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32).to(device)
        c_t2 = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32).to(device)
        
        for time_step in y.split(1, dim=1):
            # N, 1
            h_t, c_t = self.lstm1(input_t, (h_t, c_t)) # initial hidden and cell states
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2)) # new hidden and cell states
            output = self.linear(h_t2) # output from the last FC layer
            outputs.append(output)
            
        for i in range(future_preds):
            # this only generates future predictions if we pass in future_preds>0
            # mirrors the code above, using last output/prediction as input
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)
        # transform list to tensor    
        outputs = torch.cat(outputs, dim=1).to(device)
        return outputs

class MV_LSTM(torch.nn.Module):
    def __init__(self,n_features,seq_length):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 64 # number of hidden states
        self.n_layers = 2 # number of LSTM layers (stacked)
    
        self.l_lstm = torch.nn.LSTM(input_size = n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True)
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