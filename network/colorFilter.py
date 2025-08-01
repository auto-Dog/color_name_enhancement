import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init
import torchvision.transforms as transforms
# class colorFilter(nn.Module):
#     """ Kernel function: kernel(r, g, b) -> (r,g,b,rg,rb,gb,r^2,g^2,b^2,rgb,1)
#     Ref: Hong, et al., "A study of digital camera colorimetric characterization
#         based on polynomial modeling." Color Research & Application, 2001. """
#     def __init__(self):
#         super(colorFilter, self).__init__()
        
#         # Define the linear layer with 10 input features (r, g, b, rg, rb, gb, r^2, g^2, b^2, rgb) and 3 output features (r, g, b)
#         self.linear = nn.Linear(10, 3)
        
#         # Initialize the weights
#         nn.init.normal_(self.linear.weight, std=0.02)
#         # self.linear.weight.data.fill_(1e-3)  # Set all initial weights to 1e-5
#         self.linear.weight.data[0,0] = 1    # Set the weights for r, g, b to 1
#         self.linear.weight.data[1,1] = 1
#         self.linear.weight.data[2,2] = 1
#         self.sigmoid = nn.Sigmoid()
#         # Initialize the biases to zero
#         self.linear.bias.data.fill_(0)

#     def forward(self, x):
#         # Extract r, g, b channels
#         r, g, b = x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :]
        
#         # Compute the polynomial terms
#         rg = r * g
#         rb = r * b
#         gb = g * b
#         r2 = r * r
#         g2 = g * g
#         b2 = b * b
#         rgb = r * g * b
        
#         # Stack the terms along the channel dimension
#         poly_terms = torch.stack([r, g, b, rg, rb, gb, r2, g2, b2, rgb], dim=1) # B 10 H W
#         # Reshape to (batch_size, H*W, 10) for the linear layer
#         batch_size, _, H, W = x.size()
#         poly_terms = poly_terms.permute(0,2,3,1)   # B H W 10
#         poly_terms = poly_terms.view(-1,10)
        
#         # Apply the linear transformation
#         transformed = self.linear(poly_terms)   # B*H*W 3
#         # transformed = self.sigmoid(transformed)
#         # Reshape back to (batch_size, 3, H, W)
#         transformed = transformed.view(batch_size, H, W, 3)
#         transformed = transformed.permute(0,3,1,2)  # B 3 H W
        
#         return transformed

# class colorFilter(nn.Module):
#     ''' Another version, color filter based on CNN '''
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.ct_conv_1 = nn.Conv2d(3,64,(1,1))
#         self.bn1 = nn.BatchNorm2d(64)
#         self.ct_conv_2 = nn.Conv2d(64,256,(1,1))
#         self.bn2 = nn.BatchNorm2d(256)
#         self.ct_conv_3 = nn.Conv2d(256,64,(1,1))
#         self.bn3 = nn.BatchNorm2d(64)
#         self.ct_conv_4 = nn.Conv2d(64,3,(1,1))
#         self.relu = nn.ReLU()
    
#     def forward(self,x):
#         out = self.ct_conv_1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.ct_conv_2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.ct_conv_3(out)
#         out = self.bn3(out)
#         out = self.relu(out)

#         out = self.ct_conv_4(out)
#         return out
trans_compose_forward = transforms.Compose(
    [transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
)
trans_compose_reverse = transforms.Compose(
    [transforms.Normalize(mean=[-1.0,-1.0,-1.0], std=[2.0,2.0,2.0])]
)

class colorFilter(nn.Module):
    '''Color filter based on U-Net'''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        n_channels = 3
        n_out_channel = 3
        bilinear = True
        self.n_channels = n_channels
        self.n_out_channel = n_out_channel
        self.bilinear = bilinear
        factor = 2 if bilinear else 1   # when use biliner method, pre reduce the channel 

        self.inc = (DoubleConv(n_channels, 64, None, 1))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_out_channel))

    def forward(self, x):
        # accept normalized RGB input ranged (0,1) and output RGB (0,1)
        x = trans_compose_forward(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits = trans_compose_reverse(logits)
        return logits


## U-Net utils
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2, do not change (H,W)"""

    def __init__(self, in_channels, out_channels, mid_channels=None,kernel_size=3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if kernel_size == 1:
            padding = 0
        else:
            padding = kernel_size%2
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, use_dropout=True):
        super().__init__()
        self.use_dropout=use_dropout
        self.dropout = nn.Dropout(0.5)
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # only double H,w
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2] # (B,C,H,W)
        diffX = x2.size()[3] - x1.size()[3]
        # print('pad info:',diffX,diffY)  # debug

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        if self.use_dropout:
            x=self.dropout(x)
        return x


class OutConv(nn.Module):   # different from original
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        # self.conv = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1),
        #     nn.BatchNorm2d(in_channels),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # )
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.tanh(x)
        return x

if __name__ == '__main__':
    from torchsummary import summary
    # Create a dummy input tensor of shape (1, 3, H, W)
    dummy_input = torch.randn(1,3,64,64)
    # Example usage with the final corrected model
    model_final = colorFilter()
    output_final = model_final(dummy_input)
    print(output_final.shape)  # Should be (1, 3, 64, 64)
    # or use summary to check the model structure
    # summary(model_final.cuda(),(3,512,512),64)
