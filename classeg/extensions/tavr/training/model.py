import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        self.dropout_prob = 0.25
        # Encoder (downsampling path)
        self.encoder1 = self.conv_block(in_channels, 16)
        self.encoder2 = self.conv_block(16, 32)
        self.encoder3 = self.conv_block(32, 64)
        self.encoder4 = self.conv_block(64, 128)
        self.encoder5 = self.conv_block(128, 256)
        self.encoder6 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Dense layers
        #self.fc1 = nn.Linear()
        self.fc2 = nn.Linear(512, out_channels)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=self.dropout_prob),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=self.dropout_prob),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)
        enc6 = self.encoder6(enc5)

        # Bottleneck
        bottleneck = self.bottleneck(enc6)

        # Flatten bottleneck output
        
        flatten = bottleneck.view(bottleneck.size(0), -1) # [B, features]
        reduce = nn.Linear(flatten.shape[-1],512) # features -> 512
        # # Dense layers
        fc1_output = F.relu(reduce(flatten))
        output = self.fc2(fc1_output)

        return output

# Example usage
if __name__ == "__main__":
    # Example input: 1 channel, example output: 2 channels (for binary segmentation)
    model = UNet3D(in_channels=10, out_channels=2)

    # # Example input tensor
    # x = torch.randn((1, 10, 128, 128, 128))  # Batch size 1, 10 channel, 64x64x64 volume

    # # Forward pass
    # output = model(x)
    # print("Output shape:", output.shape)