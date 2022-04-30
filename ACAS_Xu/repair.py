import torch
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchnet import meter
from torchvision import transforms as T
import numpy as np
import pickle, tqdm
from simulate import ACASagent, Autoagent, env, calculate_init_bounds
from data import generate_data, sample_nocrash, ACAS_data, generate_target_data
import onnx
from onnx2pytorch import ConvertModel
import copy, time

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
        x = x - torch.Tensor([1.9791091e+04, 0.0, 0.0, 650.0, 600.0]).cuda()
        x = x / torch.Tensor([[60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]]).cuda()
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
        output = self.MatMul_H5(output)
        output = nn.ReLU(inplace=True)(output)
        output = self.MatMul_y_out(output)
        return -output
    
    def save(self, index=None):
        prefix = 'checkpoints/' + str(index) + '/model_'
        name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

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

def finetune(lr=1e-4, batch_size=16, max_epoch=60, model_index=1):
    print('-----loading model-----')
    model = read_onnx(model_index, 2)
    model = model.cuda()

    print('-----loading crash data-----')
    if model_index == 1:
        crash_data = generate_data('./results/crash_0712.pkl')
    elif model_index == 4:
        crash_data, _ = generate_target_data('./results/crash_0712.pkl')
    elif model_index == 5:
        _, crash_data = generate_target_data('./results/crash_0712.pkl')

    print('-----loading nocrash data-----')
    nocrash_data = sample_nocrash(200)
    training_set = ACAS_data(crash_data, nocrash_data, 'train')
    testing_set = ACAS_data(crash_data, nocrash_data, 'test')
    train_dataloader = DataLoader(training_set, batch_size, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(testing_set, batch_size, shuffle=True, num_workers=8)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)

    print('-----training start-----')
    acc = val(model, test_dataloader)
    print('orig acc:', acc)
    for epoch in range(max_epoch):
        for ii, (data1, label) in enumerate(train_dataloader):
            input1 = Variable(data1).cuda()
            target = Variable(label).cuda()
            optimizer.zero_grad()
            score = model(input1)
            loss = criterion(score.squeeze().squeeze(), target)
            loss.backward()
            optimizer.step()
            # if ii % 100 == 0:
            #     print(loss)
        acc = val(model, test_dataloader)
        print('epoch: ', epoch, ', accuracy: ', acc)
        model.save(model_index)

def val(model, dataloader):
    model.eval()
    confusion_matrix = meter.ConfusionMeter(5)
    loss_meter = meter.AverageValueMeter()
    for ii, data in enumerate(dataloader):
        input1, label = data
        with torch.no_grad():
            val_input1 = Variable(input1).cuda()
            val_label = Variable(label.type(torch.LongTensor)).cuda()

        score = model(val_input1)
        confusion_matrix.add(score.data.squeeze(), label.type(torch.LongTensor))

    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 0.0
    for i in range(5):
        accuracy += cm_value[i][i]
    accuracy = 100. * accuracy / float(cm_value.sum())

    return accuracy


if __name__ == '__main__':
    finetune(model_index=5)
    print('success!')