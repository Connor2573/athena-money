import torch

class model1(torch.nn.Module):
    def __init__(self, memory):
        super(model1, self).__init__()
        self.rnn1 = torch.nn.rnn() #STOPPED HERE