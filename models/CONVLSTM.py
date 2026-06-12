import torch
import torch.nn as nn
import numpy as np

from modules_convlstm import TimeDistributed, ConvLSTM


class LSTMAutoEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        time_steps: int = 8,
        hidden_dim: int = 128,  
        layers: int = 3,        
    ):
        super().__init__()

        self.time_steps = time_steps

        
        self.conv_1 = TimeDistributed(nn.Conv2d(in_channels, 32,  4, 2, 1), time_steps)
        self.norm_1 = nn.InstanceNorm3d(32,  affine=True)

        self.conv_2 = TimeDistributed(nn.Conv2d(32,  64,  4, 2, 1), time_steps)
        self.norm_2 = nn.InstanceNorm3d(64,  affine=True)

        self.conv_3 = TimeDistributed(nn.Conv2d(64,  128, 4, 2, 1), time_steps)
        self.norm_3 = nn.InstanceNorm3d(128, affine=True)

        self.conv_4 = TimeDistributed(nn.Conv2d(128, 128, 4, 2, 1), time_steps)
        self.norm_4 = nn.InstanceNorm3d(128, affine=True)

        self.conv_5 = TimeDistributed(nn.Conv2d(128, 128, 4, 2, 1), time_steps)
        self.norm_5 = nn.InstanceNorm3d(128, affine=True)

        lstm_blocks = []

        for i in range(layers):

            if i == 0:
                in_dim = 128
                out_dim = hidden_dim

            elif i == layers - 1:
                in_dim = hidden_dim
                out_dim = 128

            else:
                in_dim = hidden_dim
                out_dim = hidden_dim

            lstm_blocks.append(
                ConvLSTM(
                    input_dim=in_dim,
                    hidden_dim=[out_dim],
                    kernel_size=(3, 3),
                    num_layers=1,
                    bias=True,
                )
            )

        self.lstm_blocks = nn.ModuleList(lstm_blocks)

      
        self.deconv_1 = TimeDistributed(nn.ConvTranspose2d(128, 128, 4, 2, 1), time_steps)
        self.norm_6   = nn.InstanceNorm3d(128, affine=True)

        self.deconv_2 = TimeDistributed(nn.ConvTranspose2d(128, 128, 4, 2, 1), time_steps)
        self.norm_7   = nn.InstanceNorm3d(128, affine=True)

        self.deconv_3 = TimeDistributed(nn.ConvTranspose2d(128, 64,  4, 2, 1), time_steps)
        self.norm_8   = nn.InstanceNorm3d(64,  affine=True)

        self.deconv_4 = TimeDistributed(nn.ConvTranspose2d(64,  32,  4, 2, 1), time_steps)
        self.norm_9   = nn.InstanceNorm3d(32,  affine=True)

        self.deconv_5 = TimeDistributed(nn.ConvTranspose2d(32,  32,  4, 2, 1), time_steps)
        self.norm_10  = nn.InstanceNorm3d(32,  affine=True)

        self.conv_6 = TimeDistributed(
            nn.Conv2d(32, in_channels, 3, 1, 1), time_steps
        )

    @staticmethod
    def _norm_act(x, norm):
        x = x.transpose(1, 2)   # (B,T,C,H,W) → (B,C,T,H,W) для InstanceNorm3d
        x = norm(x)
        x = torch.relu_(x)
        return x.transpose(1, 2)

    
    def forward(self, x):

        x = self._norm_act(self.conv_1(x), self.norm_1)
        x = self._norm_act(self.conv_2(x), self.norm_2)
        x = self._norm_act(self.conv_3(x), self.norm_3)
        x = self._norm_act(self.conv_4(x), self.norm_4)
        x = self._norm_act(self.conv_5(x), self.norm_5)


        for lstm in self.lstm_blocks:
            x = lstm(x)


        x = self._norm_act(self.deconv_1(x), self.norm_6)
        x = self._norm_act(self.deconv_2(x), self.norm_7)
        x = self._norm_act(self.deconv_3(x), self.norm_8)
        x = self._norm_act(self.deconv_4(x), self.norm_9)
        x = self._norm_act(self.deconv_5(x), self.norm_10)

        return torch.sigmoid(self.conv_6(x))



if __name__ == "__main__":
    for layers in [2, 3, 4]:
        for hidden_dim in [64, 128, 256]:
            inp = torch.randn(2, 4, 1, 256, 256).cuda()
            model = LSTMAutoEncoder(
                in_channels=1,
                time_steps=4,
                hidden_dim=hidden_dim,
                layers=layers,
            ).cuda()
            out = model(inp)
            print(f"layers={layers}, hidden_dim={hidden_dim} → out={out.shape}")