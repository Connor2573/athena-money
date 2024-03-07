import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class PolicyNetwork(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(PolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(n_inputs, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, n_outputs)

        # Apply Xavier initialization
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)
    
#need to add an RNN system
    
class ComplexPolicyNetwork(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(ComplexPolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(n_inputs, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fcMid1 = nn.Linear(1024, 2048)
        self.fcMid2 = nn.Linear(2048, 4096)
        self.fcMid3 = nn.Linear(4096, 4096)
        self.fcMid4 = nn.Linear(4096, 2048)
        self.fcMid5 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, n_outputs)
        self.dropout = nn.Dropout(0.1)

        # Apply Xavier initialization
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.fc3.weight)
        init.xavier_uniform_(self.fc4.weight)
        init.xavier_uniform_(self.fc5.weight)
        init.xavier_uniform_(self.fc6.weight)
        init.xavier_uniform_(self.fcMid1.weight)
        init.xavier_uniform_(self.fcMid2.weight)
        init.xavier_uniform_(self.fcMid3.weight)
        init.xavier_uniform_(self.fcMid4.weight)
        init.xavier_uniform_(self.fcMid5.weight)

    def forward(self, x):
        x = x.unsqueeze(0)  # Add an extra dimension
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        x = F.relu(self.fc3(x))
        #x = self.dropout(x)
        x = F.relu(self.fcMid1(x))
        x = F.relu(self.fcMid2(x))
        x = F.relu(self.fcMid3(x))
        x = F.relu(self.fcMid4(x))
        x = F.relu(self.fcMid5(x))
        #x = self.dropout(x)
        x = F.relu(self.fc4(x))
        #x = self.dropout(x)
        x = F.relu(self.fc5(x))
        #x = self.dropout(x)
        return F.softmax(self.fc6(x), dim=-1).squeeze(0)