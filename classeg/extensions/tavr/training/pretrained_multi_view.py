import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.utilities.plans_handling.plans_handler import *
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

class PretrainedModelMV(nn.Module):
    def __init__(self, in_channels, out_channels, metadata_shape=21):
        super(PretrainedModelMV, self).__init__()
        self.Pretrainedencoder = torch.load("/home/student/andrewheschl/Cardiac_TAVR_Classification/modelEncoder.pt")
        # self.Pretrainedencoder = PretrainedModel.pre_encoder()
        self.conv_reduce = nn.Sequential(
            nn.Conv3d(320, 320, (3, 3, 3), (3, 1, 1), (1, 1, 1)),
            nn.LeakyReLU(),
            nn.Conv3d(320, 320, 3, (2, 2, 2), 1)
        )
        self.dropout_prob = 0.2
        self.metadata_shape = metadata_shape
        # Encoder (downsampling path)

        # self.encoder = nn.Sequential(
        #     self.conv_block(in_channels, 16),
        #     self.conv_block(16, 32),
        #     self.conv_block(32, 64),
        #     self.conv_block(64, 128),
        #     self.conv_block(128, 256),
        # )

        self.metadata_projector = nn.Linear(
            metadata_shape+320, 320
        )
        # self.fc1 = nn.Linear()
        self.classifier = nn.Sequential(
            nn.Linear(10*320, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, out_channels)
        )
        #self.freeze()
    def freeze(self):
        for param in self.Pretrainedencoder.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.Pretrainedencoder.parameters():
            param.requires_grad = True
    
    def pre_encoder():

        p = PlansManager(plans_file_or_dict="/home/student/andrewheschl/Cardiac_TAVR_Classification/plans.json")
        config = p.get_configuration(configuration_name="3d_fullres")
        architecture_class_name = config.network_arch_class_name
        arch_init_kwargs = config.network_arch_init_kwargs
        arch_init_kwargs_req_import = config.network_arch_init_kwargs_req_import
        num_input_channels = 1
        num_output_channels = 10
        model = get_network_from_plans(
                    architecture_class_name,
                    arch_init_kwargs,
                    arch_init_kwargs_req_import,
                    num_input_channels,
                    num_output_channels)
        model.load_state_dict(torch.load("/home/student/andrewheschl/Cardiac_TAVR_Classification/checkpoint_best.pth")["network_weights"])
        return model.encoder

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

    def forward(self, x, metadata=None):
        o = []
        for i in range(0,10):
            x = x[:, i:i+1, ...]
            b, t, *r = x.shape
            x = self.Pretrainedencoder(x)[-1]
            x = self.conv_reduce(x)
            flat1 = F.max_pool3d(x, kernel_size=x.size()[2:])
            flat1 = flat1.view(b, -1)
            o.append(flat1)
        o = torch.concat(o, dim=1)
        return self.classifier(o)



# Example usage
if __name__ == "__main__":
    # Example input: 1 channel, example output: 2 channels (for binary segmentation)
    model = PretrainedModel(in_channels=1, out_channels=2)

    # # Example input tensor
    x = torch.randn((2, 10, 128, 128, 128))  # Batch size 1, 10 channel, 64x64x64 volume

    # # Forward pass
    output = model(x)
    print("Output shape:", output.shape)
