from torch import nn
import torch


class Edge(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel_filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding='same',
                                      padding_mode='reflect', bias=False)

        Gx = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        Gy = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        self.sobel_filter.weight = nn.Parameter(G.unsqueeze(1), requires_grad=False)

    def forward(self, img):
        x = self.sobel_filter(img)
        x = torch.abs(x)
        return x
