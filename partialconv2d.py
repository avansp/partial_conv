###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################

# Modifications by Avan:
# 1. Removed multi-channel option
# 2. Mask is not returned. Instead it is stored under self.mask_out variable.
#    Return value from forward() is always an image.
# 3. Create additional padding that the padding will be learned by partial conv
#    So the output size will be the same if stride==1.
# 4. Mask input argument is mandatory
# 5. Removed last_size
# 6. Mask should have the same channel as the image channel.
#    If mask channel is 1, then the mask will be replicated along the channel axes.

import torch
import torch.nn.functional as F
from torch import nn, cuda
from torch.autograd import Variable

class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(PartialConv2d, self).__init__(*args, **kwargs)
        
        self.weight_maskUpdater = None 
        self.slide_winsize = None
        self.mask_out = None
        self.mask_ratio = None        
        
        # calculate new padding to learn
        # see also torch.nn.functional.pad() function
        new_pad = tuple(int((s-1)/2) for s in self.kernel_size)
        self.pconv_padding = (new_pad[0], new_pad[0], new_pad[1], new_pad[1])

    def forward(self, image_in, mask_in):
        
        # pre-checking the shapes
        im_shape = image_in.shape
        mask_shape = mask_in.shape
        
        assert len(im_shape) == 4, "Image shape must be: [BATCH, CHANNELS, HEIGHT, WIDTH]"
        assert len(mask_shape) == 4, "Mask shape must be: [BATCH, 1, HEIGHT, WIDTH]"
        assert im_shape[0]==mask_shape[0] and im_shape[2]==mask_shape[2] and im_shape[3]==mask_shape[3], \
            "Image and mask sizes do not match"
        
        # if mask channel is 1, then we'll repeat the channel
        if mask_shape[1]==1 and self.in_channels>1:
            mask_in = mask_in.repeat(1, self.in_channels, 1, 1)
        
        # pad image_in & mask_in, which will be part of the learning
        # note that mask==0 is the regions to learn for the inpainting
        image_in = F.pad(image_in, self.pconv_padding, value=0).to(self.weight.device)
        mask_in = F.pad(mask_in, self.pconv_padding, value=0).to(self.weight.device)
        
        # prepare weight for the mask
        if self.weight_maskUpdater is None:
            self.weight_maskUpdater = torch.ones(self.out_channels, 
                                                 self.in_channels, 
                                                 self.kernel_size[0], 
                                                 self.kernel_size[1]).to(mask_in)
            self.slide_winsize = self.weight_maskUpdater.shape[1] * \
                self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]
            

        with torch.no_grad():
            self.mask_out = F.conv2d(mask_in, 
                                     self.weight_maskUpdater, bias=None, 
                                     stride=self.stride, padding=self.padding, 
                                     dilation=self.dilation, groups=1)

            # for mixed precision training, change 1e-8 to 1e-6
            self.mask_ratio = self.slide_winsize/(self.mask_out + 1e-8)
            # self.mask_ratio = torch.max(self.mask_out)/(self.mask_out + 1e-8)
            self.mask_out = torch.clamp(self.mask_out, 0, 1)
            self.mask_ratio = torch.mul(self.mask_ratio, self.mask_out)

        raw_out = super(PartialConv2d, self).forward(torch.mul(image_in, mask_in))

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            image_out = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            image_out = torch.mul(image_out, self.mask_out)
        else:
            image_out = torch.mul(raw_out, self.mask_ratio)

        return image_out

    def get_output_shape(self, height, width, batch=1):
        im_in = torch.zeros(batch, self.in_channels, height, width)
        mask_in = torch.zeros(batch, 1, height, width)
        
        im_out, mask_out = self.forward(im_in,mask_in)
        
        return im_out.shape
