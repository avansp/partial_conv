#!/usr/bin/env python
# coding: utf-8

# # U-Net with Partial Convolution
# ----
# Based on: https://github.com/MathiasGruber/PConv-Keras/blob/master/notebooks/Step3%20-%20UNet%20Architecture.ipynb

# In[ ]:


from torchsummary import summary
from mask_generator import *
from partialconv2d import *
from torch import nn


# ## Encoder Block

# In[ ]:


class EncoderBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, use_batch_norm=True):
        super(EncoderBlock, self).__init__()
        
        self.pconv = PartialConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=(2,2))
        self.bn = nn.BatchNorm2d(out_channels) if use_batch_norm else None
        self.relu = nn.ReLU()
        
        # needed to save the output image for later skip connection
        self.out_image = None
        
    def forward(self, in_image, in_mask):
        self.out_image = self.pconv(in_image, in_mask)
        if self.bn is not None:
            self.out_image = self.bn(self.out_image)
        self.out_image = self.relu(self.out_image)
        
        return self.out_image
    
    def get_mask_output(self):
        return self.pconv.mask_out
    
    def get_output_shape(self, height, width, batch=1):
        img = torch.zeros(batch, self.pconv.in_channels, height, width)
        mask = torch.zeros(batch, self.pconv.in_channels, height, width)
        y = self.forward(img, mask)
        return y.size(), y.dtype, y.device


# In[ ]:


# eb = EncoderBlock(512, 512, 3).to('cuda')
# eb.get_output_shape(64, 64)


# ## Decoder Block

# In[ ]:


class DecoderBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, concat_channels, kernel_size, use_batch_norm=True):
        super(DecoderBlock, self).__init__()
        
        self.in_channels = in_channels
        self.concat_channels = concat_channels
        
        self.upsampling_img = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsampling_mask = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.pconv = PartialConv2d(
            self.in_channels + self.concat_channels, 
            out_channels, kernel_size=kernel_size, stride=1)
        self.bn = nn.BatchNorm2d(out_channels) if use_batch_norm else None
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        
    def get_mask_output(self):
        return self.pconv.mask_out
        
    def forward(self, in_image, in_mask, skip_img, skip_mask, verbose=False):
        
        # make sure all in the same devices
        if skip_img.device != in_image.device:
            skip_img = skip_img.to(in_image)
        if skip_mask.device != in_image.device:
            skip_mask = skip_mask.to(in_image)
        if in_mask.device != in_image.device:
            in_mask = in_mask.to(in_image)

        # upsampling
        up_img = self.upsampling_img(in_image)
        up_mask = self.upsampling_mask(in_mask)
        
        if verbose:
            print(f"input {in_image.size()} upsampled into {up_img.size()} + skip_img {skip_img.size()}")
            print(f"mask {in_mask.size()} upsampled into {up_mask.size()} + skip_mask {skip_mask.size()}")
            
        # partial convolution from the concatenated images & masks
        out_image = self.pconv(
            torch.cat([skip_img, up_img], dim=1),
            torch.cat([skip_mask, up_mask], dim=1)
        )
        
        # and the rest
        if self.bn is not None:
            out_image = self.bn(out_image)
        out_image = self.relu(out_image)
        
        return out_image
        
    def get_output_shape(self, height, width, batch=1):
        x = torch.zeros(batch, self.in_channels, height, width)
        y = torch.zeros(batch, self.in_channels, height, width)
        
        x_cat = torch.zeros(batch, self.concat_channels, 2*height, 2*width)
        y_cat = torch.zeros(batch, self.concat_channels, 2*height, 2*width)
        
        z = self.forward(x,y,x_cat,y_cat)
        
        return z.size(), z.dtype, z.device


# In[ ]:


# db = DecoderBlock(512, 256, 256, 3).to('cuda')
# print(f"Output shape: {db.get_output_shape(32, 32)}")


# ## Partial U-Net

# In[ ]:


class PartialUNet(nn.Module):
    
    def __init__(self, in_channels):
        super(PartialUNet, self).__init__()
        
        self.in_channels = in_channels
        
        self.encoders = nn.ModuleList([
            EncoderBlock(
                c_in, c_out, kernel_size=(ks, ks), use_batch_norm=bn
            ) for (c_in, c_out, ks, bn) in [
                (in_channels, 64, 7, False),
                (64, 128, 5, True),
                (128, 256, 5, True),
                (256, 512, 3, True),
                (512, 512, 3, True),
                (512, 512, 3, True),
                (512, 512, 3, True),
                (512, 512, 3, True)
            ]
        ])
        
        self.decoders = nn.ModuleList([
            DecoderBlock(
                c_in, c_out, c_cat, kernel_size=ks, use_batch_norm=bn
            ) for (c_in, c_out, c_cat, ks, bn) in [
                (512, 512, 512, 3, True),
                (512, 512, 512, 3, True),
                (512, 512, 512, 3, True),
                (512, 512, 512, 3, True),
                (512, 256, 256, 3, True),
                (256, 128, 128, 3, True),
                (128, 64, 64, 3, True),
                (64, 3, 3, 3, False)
            ]
        ])
        
        # last layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
    
    def forward(self, in_image, in_mask=None):
        
        # create in_mask if it is None (used in testing)
        if in_mask is None:
            in_mask = torch.ones_like(in_image).to(in_image)
        
        # encoding
        out_image = self.encoders[0](in_image, in_mask)
        for i in range(len(self.encoders)-1):
            out_image = self.encoders[i+1](out_image, self.encoders[i].get_mask_output())
            
        # decoding
        for i in range(len(self.decoders)-1):
            j = (i-2*i)-1
            out_image = self.decoders[i](
                out_image, self.encoders[j].get_mask_output(),
                self.encoders[j-1].out_image, self.encoders[j-1].get_mask_output()
            )
            
        # the last one, concate with the input image & mask
        if in_mask.shape[1]==1:
            in_mask = in_mask.repeat(1, in_image.shape[1], 1, 1)

        out_image = self.decoders[-1](
            out_image, self.encoders[0].get_mask_output(),
            in_image, in_mask
        )
            
        # last layer
        out_image = self.output_layer(out_image)

        return out_image
    
    def get_output_shape(self, height, width, batch=1):
        x = torch.zeros(batch, self.in_channels, height, width)
        m = torch.zeros(batch, 1, height, width)
        y = self.forward(x, m)
        return y.size(), y.dtype


# In[ ]:


# punet = PartialUNet(3).to('cuda')
# print(f"len(encoders) = {len(punet.encoders)}")
# print(f"len(decoders) = {len(punet.decoders)}")


# In[ ]:


# punet.get_output_shape(512, 512)


# In[ ]:




