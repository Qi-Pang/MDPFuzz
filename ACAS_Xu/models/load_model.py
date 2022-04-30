import onnx
from onnx2pytorch import ConvertModel
import torch
from torch import nn
from torchvision import transforms as T
from torch.nn import functional as F

class ACASNet(nn.Module):
    def __init__(self):
        super(ACASNet, self).__init__()
        self.MatMul_H0 = nn.Linear(5, 50, bias=True)
        self.MatMul_H1 = nn.Linear(50, 50, bias=True)
        self.MatMul_H2 = nn.Linear(50, 50, bias=True)
        self.MatMul_H3 = nn.Linear(50, 50, bias=True)
        self.MatMul_H4 = nn.Linear(50, 50, bias=True)
        self.MatMul_H5 = nn.Linear(50, 50, bias=True)
        self.MatMul_y_out = nn.Linear(50, 5, bias=True)

    def normalize(self, x):
        x = x - torch.Tensor([1.9791091e+04, 0.0, 0.0, 650.0, 600.0])
        x = x / torch.Tensor([[60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]])
        return x

    def forward(self, x):
        x = self.normalize(x)
        output = self.MatMul_H0(x)
        output = nn.ReLU(inplace=True)(output)
        output = self.MatMul_H1(output)
        output = nn.ReLU(inplace=True)(output)
        output = self.MatMul_H2(output)
        output = nn.ReLU(inplace=True)(output)
        output = self.MatMul_H3(output)
        output = nn.ReLU(inplace=True)(output)
        output = self.MatMul_H4(output)
        output = nn.ReLU(inplace=True)(output)
        active = torch.sign(output.clone())
        output = self.MatMul_H5(output)
        output = nn.ReLU(inplace=True)(output)
        output = self.MatMul_y_out(output)
        return output, active
        # return output

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))

def read_onnx(prev_a, tau):
    prev_a = int(prev_a)
    tau = int(tau)
    path_to_onnx_model = './models/ACASXU_run2a_' + str(prev_a) + '_' + str(tau) + '_batch_2000.onnx'
    onnx_model = onnx.load(path_to_onnx_model)
    pytorch_model = ConvertModel(onnx_model, experimental=True)    
    acas_xu_model = ACASNet()
    with torch.no_grad():
        for target_param, param in zip(acas_xu_model.parameters(), pytorch_model.parameters()):
            target_param.data.copy_(param.data.t())

    del pytorch_model, onnx_model

    return acas_xu_model

def load_repair(model_index=1):
    acas_xu_model = ACASNet()
    if model_index == 1:
        acas_xu_model.load('./models/1_repair.pth')
    elif model_index == 4:
        acas_xu_model.load('./models/4_repair.pth')
    elif model_index == 5:
        acas_xu_model.load('./models/5_repair.pth')

    return acas_xu_model
