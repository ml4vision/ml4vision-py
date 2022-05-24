import numpy as np
import torch
from .colors import get_colors

class Colorizer:

    def __init__(self):
        self.colors = get_colors()

    def _colorize(self, image_tensor):
        image_np = image_tensor.numpy()

        color_image = np.zeros((3, image_np.shape[0], image_np.shape[1]), np.uint8)

        unique_instances = np.unique(image_np)

        for id in unique_instances:
            id = id % 256
            mask = image_np == id

            color_image[0][mask] = self.colors[id][0]
            color_image[1][mask] = self.colors[id][1]
            color_image[2][mask] = self.colors[id][2]

        return torch.from_numpy(color_image).float()/256

    def colorize(self, image_tensor):
        image_tensor = image_tensor.squeeze()
        if image_tensor.dim() == 3:
            return torch.cat([self._colorize(image_t).unsqueeze(0) for image_t in image_tensor], 0)
        else:
            return self._colorize(image_tensor)