import torch.nn as nn
import torch
import lib.utils as utils
import numpy as np


class NeuralCDEEncoder(nn.Module):
    def __init__(self,  
                 bottleneck_function_channels,
                 ode_hidden_state_channels, 
                 learn_length_scale,
                 convcnp):
        super(NeuralCDEEncoder, self).__init__()
        self.bottleneck_function_channels = bottleneck_function_channels
        self.ode_hidden_state_channels = ode_hidden_state_channels
        self.sigma = nn.Parameter(np.log(convcnp.init_length_scale) *
                                  torch.ones(self.bottleneck_function_channels),
                                  requires_grad=learn_length_scale)
        self.sigma_fn = torch.exp
        self.initial_hidden_state_network = torch.nn.Linear(self.bottleneck_function_channels, 
                                                            self.ode_hidden_state_channels)
        utils.init_network_weights(self.initial_hidden_state_network)
        self.current_task = None
        self.convcnp = convcnp

    def forward(self, task):
        x_grid, h = self.convcnp(task['x_context'], task['y_context'])

        self.current_task = {'x': x_grid, 'y': h}

        y_initial = self.calculate_kernel(x_grid,
                                          x_grid[0, 0, 0],
                                          h).sum(axis=1)

        initial_hidden_state = self.initial_hidden_state_network(y_initial)

        return {'z0': initial_hidden_state, 't0': x_grid[0, 0, :1]}

    def derivative(self, t):
        y_derivative = self.calculate_kernel(self.current_task['x'],
                                             t, 
                                             self.current_task['y'])

        scales = self.get_scales()
        y_derivative *= (self.current_task['x'] - t) / scales ** 2

        return y_derivative.sum(axis=1)

    def calculate_kernel(self, x_grid, x, y):
        wt = self.rbf(x_grid, x)
        kernel = y * wt
        return kernel

    def get_scales(self):
        self.sigma.data.clamp_(-5, 5)
        scales = self.sigma_fn(self.sigma)[None, None, :]
        return scales

    def rbf(self, x, t):
        distances = ((x - t) ** 2)
        scales = self.get_scales()
        return torch.exp(-0.5 * distances / scales ** 2)
