"""
Deprecated model module.

Status:
- No in-repo runtime reference found in current training/testing pipeline.
- Kept temporarily for historical experiments and potential rollback.

If this module is still required by external scripts/notebooks, migrate those
callers to the current model entry in `utils.py` before removing this file.
"""

import torch
import torch.nn as nn

def weights_init(net, init_type = '', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    """
    def init_func(m):
        classname = m.__class__.__name__
        # for every Linear layer in a model
        # m.weight.data shoud be taken from a normal distribution
        # m.bias.data should be 0
        if classname.find('Linear') != -1:
            #m.weight.data.normal_(0, 0.05)
            #torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            #torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            #torch.nn.init.constant_(m.bias.data, 0.0)
            m.bias.data.fill_(0)

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

# BP Network
class Net(nn.Module):
   def __init__(self, opt):
       super(Net, self).__init__()
       self.net = nn.Sequential(
       nn.Linear(in_features=opt.in_features, out_features=50), nn.Tanh(),
       nn.Linear(50, 60), nn.Tanh(),
       nn.Linear(60, 70), nn.Tanh(),
       nn.Linear(70, 80), nn.Tanh(),
       nn.Linear(80, 90), nn.Tanh(),
       nn.Linear(90, 100), nn.Tanh(),
       nn.Linear(100, 110), nn.Tanh(),
       nn.Linear(110, 120), nn.Tanh(),
       # nn.Linear(100, 120, bias=True), nn.Tanh(),
       # nn.Linear(120, 160, bias=True), nn.Tanh(),
       # nn.Linear(160, 200, bias=True), nn.Tanh(),
       # nn.Linear(60, 60, bias=True), nn.Tanh(),
       nn.Linear(120, opt.out_features), nn.Tanh())
   def forward(self, input:torch.FloatTensor):
         return self.net(input)