"""
Deprecated model module.

Status:
- Previously reachable via `utils.create_multi_model()`.
- That entry has been removed from current in-repo pipeline.
- Kept temporarily for historical experiments and potential rollback.

If external scripts still import this module, migrate them first before
archiving or deleting this file.
"""

import torch
import torch.nn as nn

def weights_init(net, init_type = '', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
    """
    def init_func(m):
        classname = m.__class__.__name__
        # for every Linear layer in a model
        # m.weight.data shoud be taken from a normal distribution
        # m.bias.data should be 0
        if classname.find('Linear') != -1:
            m.weight.data.normal_(0, 0.05)
            #torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            #torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            #torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            #torch.nn.init.constant_(m.bias.data, 0.0)
            m.bias.data.fill_(0)

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

class BaseModel(nn.Module):
    def __init__(self, inFeat=5, outFeat=41):
        super(BaseModel, self).__init__()
        self.layer1 = nn.Linear(in_features=inFeat, out_features=100)
        self.layer2 = nn.Linear(100, 110)
        self.layer3 = nn.Linear(110, 120)
        self.layer4 = nn.Linear(120, 130)
        self.layer5 = nn.Linear(130, 140)
        self.layer6 = nn.Linear(140, 150)
        self.layer7 = nn.Linear(150, 160)
        self.layer8 = nn.Linear(160, 170)
        self.layer9 = nn.Linear(170, 180)
        self.layer10 = nn.Linear(180, 190)
        self.layer11 = nn.Linear(190, 200)
        self.layer12 = nn.Linear(200, outFeat)
        self.tanh = nn.Tanh()
        #self.dropout = nn.Dropout(p=0.5)
        # self.softmax = nn.Softmax2d(

    def forward(self, x):
        out = self.tanh(self.layer1(x))
        out = self.tanh(self.layer2(out))
        out = self.tanh(self.layer3(out))
        out = self.tanh(self.layer4(out))
        out = self.tanh(self.layer5(out))
        out = self.tanh(self.layer6(out))
        out = self.tanh(self.layer7(out))
        out = self.tanh(self.layer8(out))
        out = self.tanh(self.layer9(out))
        out = self.tanh(self.layer10(out))
        out = self.tanh(self.layer11(out))
        out = self.layer12(out)
        return out

class BaseModel1(nn.Module):
    def __init__(self, inFeat=5, outFeat=41):
        super(BaseModel1, self).__init__()
        self.layer1 = nn.Linear(in_features=inFeat, out_features=30)
        self.layer2 = nn.Linear(30, 40)
        self.layer3 = nn.Linear(40, 50)
        self.layer4 = nn.Linear(50, outFeat)
        self.tanh = nn.Tanh()
        #self.dropout = nn.Dropout(p=0.5)
        # self.softmax = nn.Softmax2d(

    def forward(self, x):
        out = self.tanh(self.layer1(x))
        out = self.tanh(self.layer2(out))
        out = self.tanh(self.layer3(out))
        out = self.tanh(self.layer4(out))
        return out

class BaseModel2(nn.Module):
    def __init__(self, inFeat=3, outFeat=6):
        super(BaseModel2, self).__init__()
        self.layer1 = nn.Linear(in_features=inFeat, out_features=50)
        self.layer2 = nn.Linear(50, 60)
        self.layer3 = nn.Linear(60, 70)
        self.layer4 = nn.Linear(70, 80)
        self.layer5 = nn.Linear(80, 90)
        self.layer6 = nn.Linear(90, outFeat)
        self.tanh = nn.Tanh()
        # self.dropout = nn.Dropout(p=0.5)
        # self.softmax = nn.Softmax2d(

    def forward(self, x):
        out = self.tanh(self.layer1(x))
        out = self.tanh(self.layer2(out))
        out = self.tanh(self.layer3(out))
        out = self.tanh(self.layer4(out))
        out = self.tanh(self.layer5(out))
        out = self.tanh(self.layer6(out))
        return out

class BaseModel3(nn.Module):
    def __init__(self, inFeat=5, outFeat=42):
        super(BaseModel3, self).__init__()
        self.layer1 = nn.Linear(in_features=inFeat, out_features=60)
        self.layer2 = nn.Linear(60, 70)
        self.layer3 = nn.Linear(70, 80)
        self.layer4 = nn.Linear(80, outFeat)
        self.tanh = nn.Tanh()
        # self.dropout = nn.Dropout(p=0.5)
        # self.softmax = nn.Softmax2d(

    def forward(self, x):
        out = self.tanh(self.layer1(x))
        out = self.tanh(self.layer2(out))
        out = self.tanh(self.layer3(out))
        out = self.layer4(out)
        return out

class BaseModel4(nn.Module):
    def __init__(self, inFeat=5, outFeat=42):
        super(BaseModel4, self).__init__()
        self.layer1 = nn.Linear(in_features=inFeat, out_features=64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 64)
        self.layer4 = nn.Linear(64, 64)
        self.layer5 = nn.Linear(64, 64)
        self.layer6 = nn.Linear(64, 64)
        self.layer7 = nn.Linear(64, outFeat)
        # self.layer7 = nn.Linear(120, 130)
        # self.layer8 = nn.Linear(130, 140)
        # self.layer9 = nn.Linear(140, outFeat)
        self.relu = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(p=0.5)
        # self.softmax = nn.Softmax2d(

    def forward(self, x):
        out = self.relu (self.layer1(x))
        out = self.relu (self.layer2(out))
        out = self.relu (self.layer3(out))
        out = self.relu (self.layer4(out))
        out = self.relu (self.layer5(out))
        out = self.relu (self.layer6(out))
        #out = self.tanh(self.layer7(out))
        #out = self.tanh(self.layer8(out))
        out = self.layer7(out)
        return out

class BaseModel5(nn.Module):
    def __init__(self, inFeat=3, outFeat=41):
        super(BaseModel5, self).__init__()
        self.layer1 = nn.Linear(in_features=inFeat, out_features=10)
        self.layer2 = nn.Linear(10, 20)
        self.layer3 = nn.Linear(20, outFeat)
        #self.layer3 = nn.Linear(10, 20)
        #self.layer4 = nn.Linear(50, outFeat)
        self.tanh = nn.Tanh()
         # self.dropout = nn.Dropout(p=0.5)
        # self.softmax = nn.Softmax2d(

    def forward(self, x):
        out = self.tanh(self.layer1(x))
        out = self.tanh(self.layer2(out))
        out = self.layer3(out)
        return out
#
#
# class BaseModel(nn.Module):
#     def __init__(self, inFeat=5, outFeat=41):
#         super(BaseModel, self).__init__()
#         self.layer1 = nn.Linear(in_features=inFeat, out_features=100)
#         self.layer2 = nn.Linear(100, 110)
#         self.layer3 = nn.Linear(110, 120)
#         self.layer4 = nn.Linear(120, 130)
#         self.layer5 = nn.Linear(130, 140)
#         self.layer6 = nn.Linear(140, 150)
#         self.layer7 = nn.Linear(150, 160)
#         self.layer8 = nn.Linear(160, 170)
#         self.layer9 = nn.Linear(170, 180)
#         self.layer10 = nn.Linear(180, 190)
#         self.layer11 = nn.Linear(190, 200)
#         self.layer12 = nn.Linear(200, outFeat)
#         self.tanh = nn.Tanh()
#         self.dropout = nn.Dropout(p=0.5)
#         # self.softmax = nn.Softmax2d(
#
#     def forward(self, x):
#         out = self.tanh(self.layer1(x))
#         out = self.tanh(self.layer2(self.dropout(out)))
#         out = self.tanh(self.layer3(self.dropout(out)))
#         out = self.tanh(self.layer4(self.dropout(out)))
#         out = self.tanh(self.layer5(self.dropout(out)))
#         out = self.tanh(self.layer6(self.dropout(out)))
#         out = self.tanh(self.layer7(self.dropout(out)))
#         out = self.tanh(self.layer8(self.dropout(out)))
#         out = self.tanh(self.layer9(self.dropout(out)))
#         out = self.tanh(self.layer10(self.dropout(out)))
#         out = self.tanh(self.layer11(self.dropout(out)))
#         out = self.layer12(self.dropout(out))
#         return out
class MutiBP(nn.Module):
    def __init__(self, opt):
       super(MutiBP, self).__init__()
       self.outputs = opt.out_features
       self.nModels = opt.nModels
       self.mubp = nn.ModuleList(
            [BaseModel4(inFeat=opt.in_features, outFeat=opt.out_features) for i in range(self.nModels)])
       # self.mubp = nn.ModuleList(
       #     [BaseModel1(inFeat=opt.in_features, outFeat=opt.out_features),BaseModel2(inFeat=opt.in_features, outFeat=opt.out_features),BaseModel3(inFeat=opt.in_features, outFeat=opt.out_features),BaseModel4(inFeat=opt.in_features, outFeat=opt.out_features)])

    def forward(self, x):
        h, w = x.shape
        models = torch.zeros((h, self.outputs, self.nModels), dtype=torch.float).cuda()
        for i in range(self.nModels):#
            input = self.mubp[i](x)
            models[:,:, i] = input
            #output = torch.sum(input, dim=0)

        output = torch.mean(models, dim=2)
        return output