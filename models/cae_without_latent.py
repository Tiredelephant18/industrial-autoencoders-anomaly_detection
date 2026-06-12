import torch
import torch.nn as nn
from collections import OrderedDict

class ConvolutionalAutoencoder(nn.Module):
   

    def __init__(self, 
                 input_channels: int = 1, 
                 kernel_size=4,
                 channels_out_1_layer = 32,
                 num_layers = 5,
                 in_size =128,
                 ):
        super(ConvolutionalAutoencoder, self).__init__()

        encode_layers = OrderedDict()
        decode_layers = OrderedDict()

        in_ch = input_channels
        out_ch = channels_out_1_layer
        img_dim = in_size

        padding = (kernel_size - 1) // 2
        if kernel_size % 2 == 0:
            output_padding = 0
        else:
            output_padding = 1

        # Encoder:
        for i in range(num_layers):
  
            encode_layers[f"conv_{i}"] = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=2, padding=padding)
            encode_layers[f"bn_{i}"] = nn.BatchNorm2d(out_ch)
            encode_layers[f"relu_{i}"] = nn.LeakyReLU(0.2, inplace=True)
            in_ch = out_ch
            out_ch = out_ch * 2
            img_dim = img_dim //2
        
        self.encoder = nn.Sequential(encode_layers)
        self.img_dim = img_dim
        self.enc_out_channels = in_ch

        
        # Decoder: 
        cur_in = out_ch//2

        for i in range(num_layers - 1):
            cur_out = cur_in //2
            decode_layers[f"conv_{i}"] = nn.ConvTranspose2d(cur_in, cur_out, kernel_size=kernel_size, stride=2, padding=padding,output_padding=output_padding )
            decode_layers[f"bn_{i}"] = nn.BatchNorm2d(cur_out)
            decode_layers[f"relu_{i}"] = nn.ReLU(inplace=True)
            cur_in  = cur_out

        decode_layers[f"conv_{num_layers-1}"] = nn.ConvTranspose2d(channels_out_1_layer, input_channels, kernel_size=kernel_size, stride=2, padding=padding,output_padding=output_padding )
        decode_layers[f"si_{num_layers-1}"] = nn.Sigmoid()
        
        self.decoder = nn.Sequential(decode_layers)

     

    def forward(self, x):
        # Encode
        latent= self.encoder(x)
        
        reconstruction = self.decoder(latent)
        return reconstruction

    def get_latent(self, x):
        x = self.encoder_conv(x)
        return self.fc_encode(x.view(x.size(0), -1))
