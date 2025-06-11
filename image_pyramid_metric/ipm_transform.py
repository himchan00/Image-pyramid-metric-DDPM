import numpy as np
import torch.nn.functional as F

class ipm_transform:
    def __init__(self):
        self.a = np.sqrt(2)/2
        self.b = 1 - self.a

    def __call__(self, img):
        if len(img.shape) == 3:
            batched = False
            img = img.unsqueeze(0)
        else:
            batched = True
        H, W = img.shape[-2:]
        assert len(img.shape) == 4, "Input must be a 4D tensor (batch_size, channels, height, width) or a 3D tensor (channels, height, width)"
        assert H % 2 == 0 and W % 2 == 0, "Image dimensions must be even for average pooling"

        avg_pooled_img = F.avg_pool2d(img, 2)
        avg_pooled_img = F.interpolate(avg_pooled_img, scale_factor=2, mode='nearest')
        out = self.a * img + self.b * avg_pooled_img

        if not batched:
            out = out.squeeze(0)
        return out


class ipm_inv_transform:
    def __init__(self):
        self.a = np.sqrt(2)
        self.b = 1 - self.a

    def __call__(self, img):
        if len(img.shape) == 3:
            batched = False
            img = img.unsqueeze(0)
        else:
            batched = True
        H, W = img.shape[-2:]
        assert len(img.shape) == 4, "Input must be a 4D tensor (batch_size, channels, height, width) or a 3D tensor (channels, height, width)"
        assert H % 2 == 0 and W % 2 == 0, "Image dimensions must be even for average pooling"

        avg_pooled_img = F.avg_pool2d(img, 2)
        avg_pooled_img = F.interpolate(avg_pooled_img, scale_factor=2, mode='nearest')
        out = self.a * img + self.b * avg_pooled_img
        
        if not batched:
            out = out.squeeze(0)
        return out
