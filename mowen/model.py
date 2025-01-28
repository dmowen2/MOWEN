import torch
import torch.nn as nn
from torchvision.models import resnet101
from timm.models.vision_transformer import VisionTransformer

class MOWEN(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, mask_ratio=0.75):
        super(MOWEN, self).__init__()

        # CNN Backbone (ResNet101 for localized features)
        self.cnn = nn.Sequential(
            *list(resnet101(pretrained=True).children())[:-2]  # Remove fully connected layers
        )

        # ViT Backbone (Vision Transformer for global relationships)
        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4,
            num_classes=embed_dim
        )

        # MAE Decoder (for pretraining)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, patch_size * patch_size)
        )

        self.mask_ratio = mask_ratio  # Percentage of patches to mask

    def forward(self, x, pretrain=False):
        # Step 1: Extract features using CNN
        cnn_features = self.cnn(x)  # Output: (B, C, H, W)

        # Step 2: Prepare patches for ViT
        patches = self.prepare_patches(cnn_features)  # Output: (B, N, patch_dim)

        # Step 3: Encode with ViT
        encoded_patches = self.vit(patches)  # Output: (B, N, embed_dim)

        if pretrain:
            # Mask patches and reconstruct (MAE)
            masked_patches = self.mask_patches(encoded_patches)
            reconstructed = self.decoder(masked_patches)
            return reconstructed
        else:
            # Return ViT-encoded features for downstream tasks
            return encoded_patches.mean(dim=1)  # Mean pooling

    def prepare_patches(self, features):
        # Convert CNN features to patches
        B, C, H, W = features.size()
        patches = features.unfold(2, 16, 16).unfold(3, 16, 16)
        patches = patches.contiguous().view(B, -1, 16 * 16)
        return patches

    def mask_patches(self, patches):
        # Randomly mask patches for MAE
        B, N, D = patches.size()
        num_masked = int(N * self.mask_ratio)
        mask = torch.rand(B, N).topk(num_masked, dim=1).indices
        patches[torch.arange(B)[:, None], mask] = 0  # Zero out masked patches
        return patches
