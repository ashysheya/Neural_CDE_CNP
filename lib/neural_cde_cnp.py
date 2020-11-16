import torch.nn as nn
import torch
import numpy as np

from lib.encoder import NeuralCDEEncoder
from lib.decoder import NeuralCDEDecoder
from lib.conv_cnp import ConvCNP
from lib.architectures import Conv


def get_net(args):
    return NeuralCDECNP(in_channels=args.in_channels,
                        conv_cnp_layers=args.conv_cnp_layers,
                        conv_cnp_channels=args.conv_cnp_channels,
                        bottleneck_function_channels=args.bottleneck_function_channels,
                        ode_hidden_state_channels=args.ode_hidden_state_channels,
                        ode_func_channels=args.ode_func_channels,
                        out_channels=args.out_channels,
                        points_per_unit=args.points_per_unit,
                        margin=args.margin,
                        receptive_field=args.receptive_field,
                        length_scale_multiplier=args.length_scale_multiplier,
                        min_x=args.min_x,
                        max_x=args.max_x)


class NeuralCDECNP(nn.Module):
    """Neural CDE CNP model."""
    def __init__(self,
                 in_channels,
                 conv_cnp_layers,
                 conv_cnp_channels,
                 bottleneck_function_channels,
                 ode_hidden_state_channels,
                 ode_func_channels,
                 out_channels,
                 points_per_unit,
                 margin,
                 receptive_field,
                 length_scale_multiplier=2.0,
                 min_x=-2.0,
                 max_x=2.0):

        super(NeuralCDECNP, self).__init__()

        architecture = Conv(receptive_field=receptive_field,
                            points_per_unit=points_per_unit,
                            num_layers=conv_cnp_layers,
                            num_channels=conv_cnp_channels)

        convcnp = ConvCNP(learn_length_scale=True,
                          points_per_unit=points_per_unit,
                          architecture=architecture,
                          margin=margin,
                          in_channels=in_channels,
                          out_channels=bottleneck_function_channels,
                          length_scale_multiplier=length_scale_multiplier,
                          min_x=min_x,
                          max_x=max_x)

        self.encoder = NeuralCDEEncoder(bottleneck_function_channels=bottleneck_function_channels,
                                        ode_hidden_state_channels=ode_hidden_state_channels,
                                        learn_length_scale=True,
                                        convcnp=convcnp)

        # Decoder specificationin_channels,

        self.decoder = NeuralCDEDecoder(in_channels=bottleneck_function_channels,
                                        hidden_channels=ode_hidden_state_channels,
                                        out_channels=out_channels,
                                        derivative=self.encoder.derivative,
                                        ode_func_channels=ode_func_channels)

    def forward(self, task):

        output = self.encoder(task)
        
        output = self.decoder(task, output)

        return output

    @property
    def num_params(self):
        """Number of parameters."""
        return np.sum([torch.tensor(param.shape).prod()
                       for param in self.parameters()])
