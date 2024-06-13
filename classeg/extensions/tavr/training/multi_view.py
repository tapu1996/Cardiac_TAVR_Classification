import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiView(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiView, self).__init__()
        self.dropout_prob = 0.25
        # Encoder (downsampling path)

        self.encoder1 = nn.Sequential(
            self.conv_block(1, 16),
            self.conv_block(16, 32),
            self.conv_block(32, 64),
            self.conv_block(64, 128),
            self.conv_block(128, 256),

        )
        self.encoder2 = nn.Sequential(
            self.conv_block(1, 16),
            self.conv_block(16, 32),
            self.conv_block(32, 64),
            self.conv_block(64, 128),
            self.conv_block(128, 256),
         
        )

        # self.fc1 = nn.Linear()
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            # nn.Linear(128, 64),
            # nn.LeakyReLU(),
            # nn.Linear(64, 32),
            # nn.LeakyReLU(),
            # nn.Dropout(0.2),
            nn.Linear(128, out_channels)
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        # Encoder
        #x = x[: (0, 4), ...]
        #x = torch.cat([x[:, 0:1, ...], x[:, 4:5, ...]], dim=1)
        x1 = x[:, 0:1, ...]
        x2 = x[:, 5:6, ...]
        #x = torch.sum(x, dim=1).unsqueeze(1)
        b, t, *r = x1.shape
        #print(x1.shape)
        #x = x.view(b*t, 1, *r) # batchify phase

        x1 = self.encoder1(x1)

        flat1 = F.max_pool3d(x1, kernel_size=x1.size()[2:])

        flat1 = flat1.view(b, -1)  # [B, features] - take phase out of batch

        b, t, *r = x2.shape
         #print(x1.shape)
        #x = x.view(b*t, 1, *r) # batchify phase

        x2 = self.encoder2(x2)

        flat2 = F.max_pool3d(x2, kernel_size=x2.size()[2:])

        flat2 = flat2.view(b, -1)  # [B, features] - take phase out of batch

        flat=  torch.cat([flat1,flat2],dim=1)

        return self.classifier(flat)
        # o = []
        # for i in range(x.shape[1]):
        #     print(i)
        #     d = x[:, i:i+1, ...]
        #     d = self.encoder(d)

        #     # Bottleneck
        #     # bottleneck = self.bottleneck(d)
        #     flat = F.max_pool3d(d, kernel_size=d.size()[2:])
        #     # Flatten bottleneck output
        #     flat = flat.view(flat.size(0), -1)  # [B, features]
        #     o.append(flat)
        # # # Dense layers
        # o = torch.concat(o, dim=1)
        # print(o.shape)
        # return self.classifier(o)


# Example usage
if __name__ == "__main__":
    # Example input: 1 channel, example output: 2 channels (for binary segmentation)
    model = ClassNetEmbedding(in_channels=1, out_channels=2)

    # # Example input tensor
    x = torch.randn((2, 10, 128, 128, 128))  # Batch size 1, 10 channel, 64x64x64 volume

    # # Forward pass
    output = model(x)
    print("Output shape:", output.shape)
