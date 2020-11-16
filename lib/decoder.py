import torch.nn as nn
import torch
import numpy as np
import controldiffeq
import lib.utils as utils


class NeuralCDEDecoder(nn.Module):
    def __init__(self, 
                 in_channels,
                 hidden_channels,
                 out_channels,  
                 derivative,
                 ode_func_channels):
        super(NeuralCDEDecoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.derivative = derivative
        self.sigma_fn = nn.Softplus()
        self.mean_layer = nn.Linear(self.hidden_channels, out_channels)
        self.sigma_layer = nn.Linear(self.hidden_channels, out_channels)
        utils.init_network_weights(self.mean_layer)
        utils.init_network_weights(self.sigma_layer)
        self.cde_func = CDEFunc(self.in_channels,
                                self.hidden_channels,
                                ode_func_channels)

    def forward(self, task, output):

        t = torch.cat([output['t0'], task['x']])

        z = controldiffeq.cdeint(dX_dt=self.derivative,
                                 z0=output['z0'],
                                 func=self.cde_func,
                                 t=t,
                                 adjoint=False, 
                                 rtol=1e-3, 
                                 atol=1e-4)[1:].permute(1, 0, 2)

        output['mean_pred'] = self.mean_layer(z)
        output['std_pred'] = self.sigma_fn(self.sigma_layer(z))

        return output

    @property
    def num_params(self):
        """Number of parameters."""
        return np.sum([torch.tensor(param.shape).prod()
                       for param in self.parameters()])


class CDEFunc(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 hidden_channels, 
                 bottleneck_channels=128):
        super(CDEFunc, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.cde_func = nn.Sequential(nn.Linear(hidden_channels, bottleneck_channels), 
                                      nn.ReLU(inplace=True),
                                      nn.Linear(bottleneck_channels, bottleneck_channels), 
                                      nn.ReLU(inplace=True),
                                      nn.Linear(bottleneck_channels, 
                                                in_channels*hidden_channels), 
                                      nn.Tanh())

    def forward(self, z):
        output = self.cde_func(z)
        output = output.view(*z.shape[:-1], 
                             self.hidden_channels, 
                             self.in_channels)
        return output
