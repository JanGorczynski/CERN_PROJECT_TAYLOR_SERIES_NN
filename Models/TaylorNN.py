import torch
import torch.nn as nn
import torch.nn.functional as F

class TaylorNN(nn.Module):
    def __init__(self,function_amount):
        super(TaylorNN, self).__init__()
        self.fc1 = nn.Linear(8, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, function_amount*2)
        self.double()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
def get_taylor_result(t,y_pred):

    n = len(y_pred)/2 
    fun_sum = torch.zeros_like(t)
    w_incr = 5000/(len(y_pred)/2)
    for i in range(0, len(y_pred), 2):
        w = w_incr* i/2
        f =  y_pred[i]/n * torch.cos(w*2 * torch.pi*t + y_pred[i+1])
        fun_sum += f
    return fun_sum