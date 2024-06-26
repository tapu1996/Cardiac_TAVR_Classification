import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassNetLSTM(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 hidden_size=512,
                 lstm_layers=1,
                 latent_size=256
                 ):
        super(ClassNetLSTM, self).__init__()
        self.dropout_prob = 0.25
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        # Encoder (downsampling path)

        self.encoder = nn.Sequential(
            self.conv_block(in_channels, 16),
            self.conv_block(16, 32),
            self.conv_block(32, 64),
            self.conv_block(64, 128),
            self.conv_block(128, latent_size)
        )

        # self.fc1 = nn.Linear()
        self.lstm = nn.LSTM(latent_size, hidden_size, lstm_layers, batch_first=True, dropout=self.dropout_prob)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LeakyReLU(),
            #nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            #nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            #nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            #nn.Dropout(0.2),
            nn.Linear(32, out_channels)
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            #nn.Dropout3d(p=self.dropout_prob),
            # nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
            # nn.Dropout3d(p=self.dropout_prob),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = torch.sum(x, dim=1).unsqueeze(1)

        # # x = x[: (0, 5), ...]
        # b, t, *r = x.shape
        # x = x.view(b*t, 1, *r) # batchify phase

        # x = self.encoder(x)

        # flat = F.max_pool3d(x, kernel_size=x.size()[2:])
        b, t, *r = x.shape
        o = []
        for i in range(x.shape[1]):
            d = x[:, i:i+1, ...]
            d = self.encoder(d)

            # Bottleneck
            # bottleneck = self.bottleneck(d)
            flat = F.max_pool3d(d, kernel_size=d.size()[2:]) #global maxpoool
            # Flatten bottleneck output
            flat = flat.view(flat.size(0), 1, -1)  # [B, features]
            o.append(flat)
        # # Dense layers
        o = torch.concat(o, dim=1) #b,t,f

        # h0 = torch.zeros(self.lstm_layers, b, self.hidden_size).to(x.device)  # (num_layers, batch, hidden_size)
        # c0 = torch.zeros(self.lstm_layers, b, self.hidden_size).to(x.device)

        # lstm_out, *_ = self.lstm(flat, (h0, c0))

        # # Only use the output of the last time step
        # last_time_step_out = lstm_out[:, -1, :]

        return self.classifier(o)


# Example usage
if __name__ == "__main__":
    # Example input: 1 channel, example output: 2 channels (for binary segmentation)
    model = ClassNetLSTM(in_channels=1, out_channels=2)

    # # Example input tensor
    x = torch.randn((1, 10, 128, 128, 128))  # Batch size 1, 10 channel, 64x64x64 volume

    # # Forward pass
    output = model(x)
    print("Output shape:", output.shape)
