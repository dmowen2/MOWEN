import torch
import torch.nn as nn
from torchvision.models import resnet101
from timm.models.vision_transformer import VisionTransformer

class CNNFeatureViT(VisionTransformer):
    def __init__(self, embed_dim=768, depth=12, num_heads=12):
        super().__init__(
            img_size=14,  # CNN already extracted 14x14 patches
            patch_size=1,  # No need for extra patching
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4,
            num_classes=embed_dim
        )

    def forward(self, x):
        # **Skip Patch Embedding**
        return self.forward_features(x)  # Directly to Transformer Blocks

class MOWEN(nn.Module):
    def __init__(self, img_size=224, embed_dim=768, depth=12, num_heads=12, mask_ratio=0.75):
        super(MOWEN, self).__init__()

        # CNN Backbone (ResNet101 for localized features)
        resnet = resnet101(weights='IMAGENET1K_V1')
        resnet.layer4[0].conv2.stride = (1, 1)  # Prevent excessive downsampling
        resnet.layer4[0].downsample[0].stride = (1, 1)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        # Linear layer to map CNN feature maps to ViT embed dim
        self.feature_projection = nn.Linear(2048, 768)  # Map CNN output to ViT embed dim


        # ViT Backbone (Vision Transformer for global relationships)
        self.vit = CNNFeatureViT(embed_dim=768, depth=12, num_heads=12)

        # MAE Decoder (for pretraining)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 768)  # Output size should match ViT embedding dim
        )


        self.mask_ratio = mask_ratio  # Percentage of patches to mask

    def forward(self, x, pretrain=False):
        cnn_features = self.cnn(x)  # (B, 2048, 14, 14)
        patches = self.prepare_patches(cnn_features)  # (B, 196, 768)
        encoded_patches = self.vit.blocks(patches)  # (B, 196, 768)
    
        if pretrain:
            # Mask patches and reconstruct (MAE)
            masked_patches = self.mask_patches(encoded_patches)

            # **Fix: Get batch size dynamically**
            B = masked_patches.shape[0]

            # Ensure decoder output matches expected embedding dimension
            decoder_out = self.decoder(masked_patches)  # Output should be (B, 196, 768)

            print(f"Decoder Output Shape Before Reshape: {decoder_out.shape}")

            expected_shape = (B, 196, 768)
            if decoder_out.shape != expected_shape:
                raise ValueError(f"Decoder output size mismatch! Expected {expected_shape}, got {decoder_out.shape}")

            return decoder_out
        else:
            return encoded_patches.mean(dim=1)  # Mean pooling




    def prepare_patches(self, features):
        B, C, H, W = features.shape  # CNN output: (B, 2048, 14, 14)

        # Reshape to (B, N, C) where N = 14 * 14
        patches = features.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)  # (B, 196, 2048)

        # Apply Linear Projection to Match ViT Embed Dim
        patches = self.feature_projection(patches)  # (B, 196, 768)

        print("Patches Shape Before ViT:", patches.shape)  # Should be (B, 196, 768)
        return patches




    def mask_patches(self, patches):
        # Randomly mask patches for MAE
        B, N, D = patches.size()
        num_masked = max(1, int(N * self.mask_ratio))  # Ensure at least 1 patch is masked
        mask = torch.rand(B, N).topk(num_masked, dim=1).indices
        patches[torch.arange(B)[:, None], mask] = 0  # Zero out masked patches
        return patches
