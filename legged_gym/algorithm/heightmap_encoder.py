# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class HeightmapEncoder(nn.Module):
    """
    MLP-based encoder for GT heightmap data that compresses terrain height information
    from sampled points around the robot into a compact representation for critic privileged information.
    Only processes clean GT heightmap data, no noise simulation.
    """
    is_heightmap_encoder = True
    is_vae = False

    def __init__(
        self,
        num_input_dim,  # Number of height points (same as MLP_Encoder pattern)
        num_output_dim,  # Dimension of compressed representation
        hidden_dims=[256, 256],  # Hidden dimensions for MLP layers (same as MLP_Encoder default)
        activation="elu",
        orthogonal_init=False,
        output_detach=False,
        **kwargs,
    ):
        if kwargs:
            print(
                "HeightmapEncoder.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super(HeightmapEncoder, self).__init__()

        self.orthogonal_init = orthogonal_init
        self.output_detach = output_detach
        self.num_input_dim = num_input_dim
        self.num_output_dim = num_output_dim

        activation = get_activation(activation)

        if num_output_dim > 0:
            # Encoder - EXACTLY same structure as MLP_Encoder
            encoder_layers = []
            encoder_layers.append(nn.Linear(num_input_dim, hidden_dims[0]))
            if self.orthogonal_init:
                torch.nn.init.orthogonal_(encoder_layers[-1].weight, np.sqrt(2))
            encoder_layers.append(activation)
            for l in range(len(hidden_dims)):
                if l == len(hidden_dims) - 1:
                    encoder_layers.append(nn.Linear(hidden_dims[l], num_output_dim))
                    if self.orthogonal_init:
                        torch.nn.init.orthogonal_(encoder_layers[-1].weight, 0.01)
                        torch.nn.init.constant_(encoder_layers[-1].bias, 0.0)
                else:
                    encoder_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                    if self.orthogonal_init:
                        torch.nn.init.orthogonal_(encoder_layers[-1].weight, np.sqrt(2))
                        torch.nn.init.constant_(encoder_layers[-1].bias, 0.0)
                    encoder_layers.append(activation)
            self.encoder = nn.Sequential(*encoder_layers)
            # Decoder for autoencoder auxiliary loss
            decoder_layers = []
            decoder_layers.append(nn.Linear(num_output_dim, hidden_dims[-1]))
            decoder_layers.append(activation)
            for l in reversed(range(len(hidden_dims))):
                if l == 0:
                    decoder_layers.append(nn.Linear(hidden_dims[l], self.num_input_dim))
                else:
                    decoder_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l - 1]))
                    decoder_layers.append(activation)
            self.decoder = nn.Sequential(*decoder_layers)
        else:
            # Disabled encoder - create dummy network
            self.encoder = nn.Identity()
            self.decoder = nn.Identity()

        print(f"HeightmapEncoder MLP: {self.encoder}")

        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def forward(self, input):
        if self.num_output_dim == 0:
            return torch.zeros((input.shape[0], 0), device=input.device, dtype=input.dtype)
        # Normalize input for stability
        input_normalized = self._normalize_input(input)
        return self.encoder(input_normalized)

    def encode(self, input):
        if self.num_output_dim == 0:
            self.encoder_out = torch.zeros((input.shape[0], 0), device=input.device, dtype=input.dtype)
            return self.encoder_out
        # Normalize input for stability
        input_normalized = self._normalize_input(input)
        self.encoder_out = self.encoder(input_normalized)
        if self.output_detach:
            return self.encoder_out.detach()
        else:
            return self.encoder_out
    
    # def _normalize_input(self, input):
    #     """Simple normalization for GT heightmap data"""
    #     # Simple normalization for clean GT data
    #     mean = input.mean()
    #     std = input.std() + 1e-6
        
    #     normalized = (input - mean) / std
    #     # Gentle clamping to preserve GT information
    #     normalized = torch.clamp(normalized, -3.0, 3.0)
    #     return normalized
    def _normalize_input(self, input):
        return input

    def get_encoder_out(self):
        return self.encoder_out

    def inference(self, input):
        with torch.no_grad():
            return self.encoder(input)
    
    # Auxiliary helpers for autoencoder
    def decode(self, z):
        return self.decoder(z)

    def forward_aux(self, input):
        z = self.encode(input)
        x_hat = self.decode(z)
        return z, x_hat
    


def get_activation(act_name):
    """Get activation function by name"""
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None

# Backwards-compatibility alias to match config name
# This lets eval("Heightmap_Encoder") resolve without changing configs
Heightmap_Encoder = HeightmapEncoder