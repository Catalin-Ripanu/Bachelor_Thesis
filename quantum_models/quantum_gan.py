from typing import Literal, Callable, Optional
import torch
import torch.nn as nn
from diff_aug import DiffAugment

from quantum_transformer import (
    TransformerEncoder,
    TransformerRK4,
    TransformerRK4Enhanced
)


class ImgPatches(nn.Module):
    def __init__(self, input_channel=3, dim=768, patch_size=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            input_channel, dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, img):
        patches = self.patch_embed(img).flatten(2).transpose(1, 2)
        return patches


def UpSampling(x, H, W):
    B, N, C = x.size()
    assert N == H * W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H * W)
    x = x.permute(0, 2, 1)
    return x, H, W


class Generator(nn.Module):
    """docstring for Generator"""

    def __init__(
        self,
        depth1=5,
        depth2=4,
        depth3=2,
        initial_size=8,
        dim=384,
        heads=4,
        mlp_ratio=4,
        drop_rate=0.0,
        quantum_attn_circuit=None,
        quantum_mlp_circuit=None,
    ):  # ,device=device):
        super(Generator, self).__init__()

        self.quantum_attn_circuit = quantum_attn_circuit
        self.quantum_mlp_circuit = quantum_mlp_circuit

        # self.device = device
        self.initial_size = initial_size
        self.dim = dim
        self.depth1 = depth1
        self.depth2 = depth2
        self.depth3 = depth3
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.droprate_rate = drop_rate

        self.mlp = nn.Linear(1024, (self.initial_size**2) * self.dim)

        self.positional_embedding_1 = nn.Parameter(torch.zeros(1, (8**2), 384))
        self.positional_embedding_2 = nn.Parameter(
            torch.zeros(1, (8 * 2) ** 2, 384 // 4)
        )
        self.positional_embedding_3 = nn.Parameter(
            torch.zeros(1, (8 * 4) ** 2, 384 // 16)
        )
        self.positional_embedding_3 = nn.Parameter(
            torch.zeros(1, (8 * 4) ** 2, 384 // 32)
        )

        self.TransformerEncoder_encoder1 = [TransformerEncoder(
            hidden_size=self.dim,
            num_heads=self.heads,
            mlp_hidden_size=self.mlp_ratio,
            dropout=self.droprate_rate,
            quantum_attn_circuit=self.quantum_attn_circuit,
            quantum_mlp_circuit=self.quantum_mlp_circuit,
        )for _ in range(self.depth1)]

        self.TransformerEncoder_encoder2 = [TransformerEncoder(
            hidden_size=self.dim//4,
            num_heads=self.heads,
            mlp_hidden_size=self.mlp_ratio,
            dropout=self.droprate_rate,
            quantum_attn_circuit=self.quantum_attn_circuit,
            quantum_mlp_circuit=self.quantum_mlp_circuit,
        )for _ in range(self.depth2)]

        self.TransformerEncoder_encoder3 = [TransformerEncoder(
            hidden_size=self.dim//16,
            num_heads=self.heads,
            mlp_hidden_size=self.mlp_ratio,
            dropout=self.droprate_rate,
            quantum_attn_circuit=self.quantum_attn_circuit,
            quantum_mlp_circuit=self.quantum_mlp_circuit,
        )for _ in range(self.depth3)]

        self.TransformerEncoder_encoder4 = TransformerRK4Enhanced(
            hidden_size=self.dim//32,
            num_heads=self.heads,
            mlp_hidden_size=self.mlp_ratio,
            dropout=self.droprate_rate,
            quantum_attn_circuit=self.quantum_attn_circuit,
            quantum_mlp_circuit=self.quantum_mlp_circuit,
        )

        self.linear = nn.Sequential(nn.Conv2d(self.dim // 32, 3, 1, 1, 0))

    def forward(self, noise):

        x = self.mlp(noise).view(-1, self.initial_size**2, self.dim)

        x = x + self.positional_embedding_1
        H, W = self.initial_size, self.initial_size

        for block in self.TransformerEncoder_encoder1:
            x = block(x, dim=self.dim)

        x, H, W = UpSampling(x, H, W)
        x = x + self.positional_embedding_2

        for block in self.TransformerEncoder_encoder2:
            x = block(x, dim=self.dim//4)

        x, H, W = UpSampling(x, H, W)
        x = x + self.positional_embedding_3
        
        for block in self.TransformerEncoder_encoder3:
            x = block(x, dim=self.dim//16)

        x, H, W = UpSampling(x, H, W)
        x = x + self.positional_embedding_4

        x = self.TransformerEncoder_encoder4(x, dim=self.dim // 32)
        x = self.linear(x.permute(0, 2, 1).view(-1, self.dim // 32, H, W))

        return x


class Discriminator(nn.Module):
    def __init__(
        self,
        diff_aug,
        image_size=32,
        patch_size=4,
        input_channel=3,
        num_classes=1,
        dim=384,
        depth=7,
        heads=4,
        mlp_ratio=4,
        drop_rate=0.0,
        quantum_attn_circuit=None,
        quantum_mlp_circuit=None,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("Image size must be divisible by patch size.")
        self.quantum_attn_circuit = quantum_attn_circuit
        self.quantum_mlp_circuit = quantum_mlp_circuit
        num_patches = (image_size // patch_size) ** 2
        self.diff_aug = diff_aug
        self.patch_size = patch_size
        self.depth = depth
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.dim = dim
        self.drop_rate = drop_rate
        # Image patches and embedding layer
        self.patches = ImgPatches(input_channel, dim, self.patch_size)

        # Embedding for patch position and class
        self.positional_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.class_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.positional_embedding, std=0.2)
        nn.init.trunc_normal_(self.class_embedding, std=0.2)

        self.droprate = nn.Dropout(p=drop_rate)

        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = DiffAugment(x, self.diff_aug)
        b = x.shape[0]
        cls_token = self.class_embedding.expand(b, -1, -1)

        x = self.patches(x)
        x = torch.cat((cls_token, x), dim=1)
        x += self.positional_embedding
        x = self.droprate(x)
        for block in range(self.depth):
            x = TransformerEncoder(
            hidden_size=self.dim,
            num_heads=self.heads,
            mlp_hidden_size=self.mlp_ratio,
            dropout=self.drop_rate,
            quantum_attn_circuit=self.quantum_attn_circuit,
            quantum_mlp_circuit=self.quantum_mlp_circuit,
        )(x, dim=self.dim)
        x = self.norm(x)
        x = self.out(x[:, 0])
        return x
