"""
Neural network models for Bootstrapped Deep Q-Network (DQN) with optional dueling architecture and randomized priors.
Defines core convolutional layers, value/advantage heads, ensemble wrapper, and prior integration.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython import embed

# custom weight initialization for Linear and Conv2d modules
# keeps distributions centered for stable training

# from the DQN paper
#The first convolution layer convolves the input with 32 filters of size 8 (stride 4),
#the second layer has 64 layers of size 4
#(stride 2), the final convolution layer has 64 filters of size 3 (stride
#1). This is followed by a fully-connected hidden layer of 512 units.

# Initialize fully-connected or convolutional layer weights/biases
def weights_init(m):
    """Apply custom initialization to model layers"""
    classtype = m.__class__
    if classtype == nn.Linear or classtype == nn.Conv2d:
        print("default init")
        #m.weight.data.normal_(0.0, 0.02)
        #m.bias.data.fill_(0)
    elif classtype == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    else:
        print('%s is not initialized.' %classtype)

# Feature extractor from stacked frames, replicates DQN paper conv layers
class CoreNet(nn.Module):
    """Convolutional network to process state frames into feature embeddings"""
    def __init__(self, network_output_size=84, num_channels=4):
        super(CoreNet, self).__init__()
        self.network_output_size = network_output_size
        self.num_channels = num_channels
        # params from ddqn appendix
        self.conv1 = nn.Conv2d(self.num_channels, 32, 8, 4)
        # TODO - should we have this init during PRIOR code?
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.conv1.apply(weights_init)
        self.conv2.apply(weights_init)
        self.conv3.apply(weights_init)

    def forward(self, x):
        # apply 3 conv layers with relu activations and flatten
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # size after conv3
        reshape = 64*7*7
        x = x.view(-1, reshape)
        return x

# Optional dueling network head splitting into value and advantage streams
class DuelingHeadNet(nn.Module):
    """Dueling DQN head: outputs Q(s,a) by combining value and advantage streams"""
    def __init__(self, n_actions=4):
        super(DuelingHeadNet, self).__init__()
        mult = 64*7*7
        self.split_size = 512
        self.fc1 = nn.Linear(mult, self.split_size*2)
        self.value = nn.Linear(self.split_size, 1)
        self.advantage = nn.Linear(self.split_size, n_actions)
        self.fc1.apply(weights_init)
        self.value.apply(weights_init)
        self.advantage.apply(weights_init)

    def forward(self, x):
        # split features, compute value and advantage, then combine
        x1,x2 = torch.split(F.relu(self.fc1(x)), self.split_size, dim=1)
        value = self.value(x1)
        advantage = self.advantage(x2)
        # value is shape [batch_size, 1]
        # advantage is shape [batch_size, n_actions]
        q = value + torch.sub(advantage, torch.mean(advantage, dim=1, keepdim=True))
        return q

# Standard Q-value head without dueling
class HeadNet(nn.Module):
    """Feed-forward head producing Q-values for each action"""
    def __init__(self, n_actions=4):
        super(HeadNet, self).__init__()
        mult = 64*7*7
        self.fc1 = nn.Linear(mult, 512)
        self.fc2 = nn.Linear(512, n_actions)
        self.fc1.apply(weights_init)
        self.fc2.apply(weights_init)

    def forward(self, x):
        # fully-connected layers to produce raw Q-values
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Ensemble network combining a shared core and multiple heads for bootstrapping
class EnsembleNet(nn.Module):
    """Bootstrapped ensemble of DQN heads sharing a convolutional core"""
    def __init__(self, n_ensemble, n_actions, network_output_size, num_channels, dueling=False):
        super(EnsembleNet, self).__init__()
        self.core_net = CoreNet(network_output_size=network_output_size, num_channels=num_channels)
        self.dueling = dueling
        if self.dueling:
            print("using dueling dqn")
            self.net_list = nn.ModuleList([DuelingHeadNet(n_actions=n_actions) for k in range(n_ensemble)])
        else:
            self.net_list = nn.ModuleList([HeadNet(n_actions=n_actions) for k in range(n_ensemble)])

    def _core(self, x):
        # shared feature extractor
        return self.core_net(x)

    def _heads(self, x):
        # apply each head to extracted features
        return [net(x) for net in self.net_list]

    def forward(self, x, k):
        # if k specified, return single head output; otherwise all heads
        if k is not None:
            return self.net_list[k](self.core_net(x))
        else:
            core_cache = self._core(x)
            net_heads = self._heads(core_cache)
            return net_heads

# Wrapper to integrate a randomized prior network into ensemble predictions
class NetWithPrior(nn.Module):
    """Adds a fixed prior network to each ensemble head output"""
    def __init__(self, net, prior, prior_scale=1.):
        super(NetWithPrior, self).__init__()
        self.net = net
        # used when scaling core net
        self.core_net = self.net.core_net
        self.prior_scale = prior_scale
        if self.prior_scale > 0.:
            self.prior = prior

    def forward(self, x, k):
        # compute network output with optional prior scaling
        if hasattr(self.net, "net_list"):
            if k is not None:
                if self.prior_scale > 0.:
                    return self.net(x, k) + self.prior_scale * self.prior(x, k).detach()
                else:
                    return self.net(x, k)
            else:
                core_cache = self.net._core(x)
                net_heads = self.net._heads(core_cache)
                if self.prior_scale <= 0.:
                    return net_heads
                else:
                    prior_core_cache = self.prior._core(x)
                    prior_heads = self.prior._heads(prior_core_cache)
                    return [n + self.prior_scale * p.detach() for n, p in zip(net_heads, prior_heads)]
        else:
            raise ValueError("Only works with a net_list model")

    def return_prior(self, x, k):
        # return only the prior component for inspection
        if hasattr(self.net, "net_list"):
            if k is not None:
                if self.prior_scale > 0.:
                    return self.prior_scale * self.prior(x, k)
                else:
                    return self.prior(x, k)
            else:
                core_cache = self.prior._core(x)
                net_heads = self.prior._heads(core_cache)
                if self.prior_scale <= 0.:
                    return net_heads
                else:
                    prior_core_cache = self.prior._core(x)
                    prior_heads = self.prior._heads(prior_core_cache)
                    return [self.prior_scale * p for p in prior_heads]
        else:
            raise ValueError("Only works with a net_list model")

# w = torch.empty(3, 5)
# nn.init.normal_(w, 0,0.2)
# print(w.size(-1))