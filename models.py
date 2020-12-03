

import torch
import numpy as np
import torch.nn as nn
from torchvision.models import vgg19


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        vgg19_model = vgg19(pretrained=True)

        # Extracts features at the 11th layer
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:12])

    def forward(self, img):
        out = self.feature_extractor(img)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.Conv2d(in_features, in_features, 3, 1, 1),
                        nn.BatchNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_features, in_features, 3, 1, 1),
                        nn.BatchNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super(GeneratorResNet, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 9, 1, 4),
            nn.ReLU(inplace=True)
        )

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(32))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32))

        # Upsampling layers
        upsampling = []
        for out_features in range(2):
            upsampling += [ nn.Conv2d(32, 128, 3, 1, 1),
                            nn.BatchNorm2d(128),
                            nn.PixelShuffle(upscale_factor=2),
                            nn.ReLU(inplace=True)]
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Sequential(nn.Conv2d(32, out_channels, 9, 1, 4), nn.Sigmoid())

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for out_filters, stride, normalize in [ (64, 1, False),
                                                (64, 2, True),
                                                (128, 1, True),
                                                (128, 2, True),
                                                (256, 1, True),
                                                (256, 2, True),
                                                (512, 1, True),
                                                (512, 2, True),]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        # Output layer
        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)
    
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc = nn.Linear(40, 512)
        self._enc_mu = nn.Linear(512, 1024)
        self._enc_log_sigma = nn.Linear(512, 1024)

    def _sample_latent(self, attrs):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        h_enc = self.fc(attrs)
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma*std_z  # Reparameterization trick

    def forward(self, attrs):
        z = self._sample_latent(attrs)
        return z.view(-1,1,32,32)
    
class Modified_Generator(nn.Module):
    def __init__(self):
        super(Modified_Generator, self).__init__()
        self.generated_output = GeneratorResNet(in_channels=1,  n_residual_blocks=24)        
        
    def forward(self, input):
        output = self.generated_output(input)        
        return output
    
class GenerativeModel(nn.Module):
    def __init__(self):
        super(GenerativeModel, self).__init__()
        self.vae = VAE()
        self.generator = Modified_Generator()
    
    def forward(self, input):
        latent_repr = self.vae(input)
        output = self.generator(latent_repr)
        return output
    

