from math import sqrt
from itertools import product
import torch
import numpy as np
from typing import List


class PriorBox(object):

    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size: int = cfg['min_dim']
        self.num_priors: int = len(cfg['aspect_ratios'])
        self.variance: List[float] = cfg['variance'] or [0.1]
        self.feature_maps: List[int] = cfg['feature_maps']
        self.min_sizes: List[int] = cfg['min_sizes']
        self.max_sizes: List[int] = cfg['max_sizes']
        self.steps: List[int] = cfg['steps']
        self.aspect_ratios: List[List[int]] = cfg['aspect_ratios']
        self.clip: bool = cfg['clip']
        self.version: str = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean: List[float] = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k: float = self.image_size / self.steps[k]
                cx: float = (j + 0.5) / f_k
                cy: float = (i + 0.5) / f_k

                s_k: float = self.min_sizes[k]/self.image_size
                mean.extend([cx, cy, s_k, s_k])

                if self.max_sizes:
                    s_k_prime: float = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                    mean.extend([cx, cy, s_k_prime, s_k_prime])

                for ar in self.aspect_ratios[k]:
                    mean.extend([cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)])
                    mean.extend([cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)])

        output = torch.from_numpy(np.array(mean, dtype=np.float32)).cuda().view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
