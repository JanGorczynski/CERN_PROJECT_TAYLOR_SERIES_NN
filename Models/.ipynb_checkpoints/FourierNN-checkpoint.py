import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierNN(nn.Module):
    def __init__(self,function_amount):
        super(FourierNN, self).__init__()
        self.function_amount = function_amount
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
        x = x.view(2 ,self.function_amount )
        return x
    
def get_fourier_result(t,y_pred):
    amplitudes = y_pred[0]
    phases = y_pred[1]
    n = len(amplitudes)
    fun_sum = torch.zeros_like(t)
    w_incr = 5000/(n)
    for i in range(len(amplitudes)):
        w = w_incr* i
        f =  amplitudes[i] * torch.cos(2*torch.pi*w*t + phases[i])
        fun_sum += f
    return fun_sum