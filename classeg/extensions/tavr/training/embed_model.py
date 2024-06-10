import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassNetEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ClassNetEmbedding, self).__init__()
        self.dropout_prob = 0.25
        # Encoder (downsampling path)

        self.encoder = nn.Sequential(
            self.conv_block(in_channels, 16),
            self.conv_block(16, 64),
            self.conv_block(64, 128),
            self.conv_block(128, 256),
            self.conv_block(256, 512)
        )

        # self.fc1 = nn.Linear()
        self.classifier = nn.Sequential(
            nn.Conv1d(512*10, 256, kernel_size=3),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, out_channels)
        )

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
        b, t, *r = x.shape
        x = x.view(b*t, 1, *r) # batchify phase

        x = self.encoder(x)

        flat = F.max_pool3d(x, kernel_size=x.size()[2:])

        flat = flat.view(b, -1)  # [B, features] - take phase out of batch

        return self.classifier(flat)


# Example usage
if __name__ == "__main__":
    # Example input: 1 channel, example output: 2 channels (for binary segmentation)
    model = ClassNetEmbedding(in_channels=1, out_channels=2)

    # # Example input tensor
    x = torch.randn((2, 10, 128, 128, 128))  # Batch size 1, 10 channel, 64x64x64 volume

    # # Forward pass
    output = model(x)
    print("Output shape:", output.shape)
