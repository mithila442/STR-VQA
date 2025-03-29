import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetWithRegisterTokens(nn.Module):
    def __init__(self, in_channels, out_channels=1, num_register_tokens=4, token_dim=64):
        super(UNetWithRegisterTokens, self).__init__()

        self.num_register_tokens = num_register_tokens
        self.token_dim = token_dim

        # Learnable Register Tokens
        self.register_tokens = nn.Parameter(torch.randn(1, num_register_tokens, token_dim, 1, 1))

        # Initial Feature Projection for Tokens
        self.token_proj = nn.Conv3d(token_dim, num_register_tokens, kernel_size=1)

        # Encoder
        self.encoder1 = self.conv_block(in_channels + num_register_tokens, 64, is_3d=True)
        self.encoder2 = self.conv_block(64, 128, is_3d=True)
        self.encoder3 = self.conv_block(128, 256, is_3d=True)
        self.encoder4 = self.conv_block(256, 512, is_3d=True)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024, is_3d=True)

        # Spatial Attention (Same as models1.py)
        self.attention = nn.Sequential(
            nn.Conv3d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(1024, 1024, kernel_size=1),
            nn.Sigmoid()
        )

        # Decoder
        self.decoder4 = self.conv_block(1024 + 512, 512, is_3d=True)
        self.decoder3 = self.conv_block(512 + 256, 256, is_3d=True)
        self.decoder2 = self.conv_block(256 + 128, 128, is_3d=True)
        self.decoder1 = self.conv_block(128 + 64, 64, is_3d=True)

        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels, is_3d=False):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()

        x = x.reshape(batch_size, channels, seq_len, height, width)

        # Expand Register Tokens for Batch Size
        register_tokens = self.register_tokens.repeat(batch_size, 1, 1, 1, 1)  # (B, 4, 64, 1, 1)
        register_tokens = register_tokens.permute(0, 2, 1, 3, 4)  # (B, 64, 4, 1, 1)

        # Apply Conv3D Projection
        register_tokens = self.token_proj(register_tokens)
        register_tokens = F.interpolate(register_tokens, size=(x.shape[2], x.shape[3], x.shape[4]), mode='nearest')

        # Concatenate Register Tokens to Input
        x = torch.cat([register_tokens, x], dim=1)  # (B, 7, T, H, W)

        # Encoder Path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        # Bottleneck with Spatial Attention
        bottleneck = self.bottleneck(self.pool(enc4))
        attention = self.attention(bottleneck)
        bottleneck = bottleneck * attention

        # Decoder Path (With Spatial Matching)
        dec4 = self.decoder4(torch.cat([F.adaptive_avg_pool3d(bottleneck, output_size=enc4.shape[2:]), enc4], dim=1))
        dec3 = self.decoder3(torch.cat([F.adaptive_avg_pool3d(dec4, output_size=enc3.shape[2:]), enc3], dim=1))
        dec2 = self.decoder2(torch.cat([F.adaptive_avg_pool3d(dec3, output_size=enc2.shape[2:]), enc2], dim=1))
        dec1 = self.decoder1(torch.cat([F.adaptive_avg_pool3d(dec2, output_size=enc1.shape[2:]), enc1], dim=1))

        # Final Output with Spatial Matching
        return self.final_conv(F.adaptive_avg_pool3d(dec1, output_size=(seq_len, height, width)))