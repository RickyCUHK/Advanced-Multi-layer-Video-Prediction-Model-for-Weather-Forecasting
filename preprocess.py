
import torch
import numpy as np

def reshape_patch(img_tensor, patch_size):
    img_tensor = img_tensor.permute([0,1,3,4,2])
    batch_size, seq_length, img_height, img_width, num_channels = img_tensor.shape
    img_tensor = img_tensor.reshape(batch_size, seq_length, img_height//patch_size, patch_size,
                           img_width//patch_size, patch_size, num_channels)
    img_tensor = img_tensor.permute([0, 1, 2, 4, 3, 5, 6])
    patch_tensor = img_tensor.reshape(batch_size, seq_length,
                                      img_height//patch_size, img_width//patch_size,
                                      patch_size*patch_size*num_channels)
    patch_tensor = patch_tensor.permute(0, 1, 4, 2, 3)
    return patch_tensor

def reshape_patch_back(patch_tensor, patch_size):
    patch_tensor = patch_tensor.permute([0, 1, 3, 4, 2])
    batch_size, seq_length, patch_height, patch_width, channels = patch_tensor.shape
    img_channels = channels // (patch_size*patch_size)
    patch_tensor = patch_tensor.reshape(batch_size, seq_length,
                                        patch_height, patch_width,
                                        patch_size, patch_size,
                                        img_channels)
    patch_tensor = patch_tensor.permute([0, 1, 2, 4, 3, 5, 6])
    img_tensor = patch_tensor.reshape(batch_size, seq_length,
                                      patch_height * patch_size, patch_width * patch_size,
                                      img_channels)
    img_tensor = img_tensor.permute(0, 1, 4, 2, 3)
    return img_tensor

def Change_color(gray):
    mask = (gray < 1.0) + 0
    gray = 255 * mask + (1 - mask) * gray
    r = gray.copy()
    g = gray.copy()
    b = gray.copy()
    mask = (gray <= 10) * (gray >= 0)
    r = mask * 255 + (1 - mask) * r
    g = mask * 245 + (1 - mask) * g
    b = 0 * mask + (1 - mask) * b

    mask = (gray <= 15) * (gray >= 10)
    r = mask * 255 + (1 - mask) * r
    g = mask * 191 + (1 - mask) * g
    b = mask * 0 + (1 - mask) * b

    mask = (gray <= 20) * (gray >= 15)
    r = mask * 255 + (1 - mask) * r
    g = mask * 0 + (1 - mask) * g
    b = mask * 0 + (1 - mask) * b

    mask = (gray <= 25) * (gray >= 20)
    r = mask * 0 + (1 - mask) * r
    g = mask * 255 + (1 - mask) * g
    b = mask * 0 + (1 - mask) * b

    mask = (gray <= 30) * (gray >= 25)
    r = mask * 0 + (1 - mask) * r
    g = mask * 205 + (1 - mask) * g
    b = mask * 0 + (1 - mask) * b

    mask = (gray <= 35) * (gray >= 30)
    r = mask * 0 + (1 - mask) * r
    g = mask * 139 + (1 - mask) * g
    b = mask * 0 + (1 - mask) * b

    mask = (gray <= 40) * (gray >= 35)
    r = mask * 0 + (1 - mask) * r
    g = mask * 255 + (1 - mask) * g
    b = mask * 255 + (1 - mask) * b

    mask = (gray <= 45) * (gray >= 40)
    r = mask * 0 + (1 - mask) * r
    g = mask * 215 + (1 - mask) * g
    b = mask * 255 + (1 - mask) * b

    mask = (gray <= 50) * (gray >= 45)
    r = mask * 63 + (1 - mask) * r
    g = mask * 133 + (1 - mask) * g
    b = mask * 205 + (1 - mask) * b

    mask = (gray <= 55) * (gray >= 50)
    r = mask * 0 + (1 - mask) * r
    g = mask * 0 + (1 - mask) * g
    b = mask * 255 + (1 - mask) * b

    mask = (gray <= 60) * (gray >= 55)
    r = mask * 34 + (1 - mask) * r
    g = mask * 34 + (1 - mask) * g
    b = mask * 178 + (1 - mask) * b

    mask = (gray <= 65) * (gray >= 60)
    r = mask * 0 + (1 - mask) * r
    g = mask * 0 + (1 - mask) * g
    b = mask * 139 + (1 - mask) * b

    mask = (gray <= 70) * (gray >= 65)
    r = mask * 255 + (1 - mask) * r
    g = mask * 0 + (1 - mask) * g
    b = mask * 255 + (1 - mask) * b

    mask = (gray <= 75) * (gray >= 70)
    r = mask * 255 + (1 - mask) * r
    g = mask * 48 + (1 - mask) * g
    b = mask * 155 + (1 - mask) * b

    mask = (gray > 75)
    r = mask * 250 + (1 - mask) * r
    g = mask * 250 + (1 - mask) * g
    b = mask * 255 + (1 - mask) * b

    rgb = np.concatenate([r, g, b], axis=4)
    return rgb
