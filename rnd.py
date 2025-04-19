import torch
import torch.nn as nn
import torch.nn.functional as F


class RNDTarget(nn.Module):
    """
    Fixed (frozen) random network for RND.  Its parameters are initialized randomly and never updated.
    Takes input states of shape [B, C, H, W] and outputs embeddings of dim rep_size.
    """
    def __init__(self, input_channels=4, convfeat=32, rep_size=512):
        super(RNDTarget, self).__init__()
        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(input_channels, convfeat, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(convfeat, convfeat * 2, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(convfeat * 2, convfeat * 2, kernel_size=3, stride=1)
        # Final linear projection to rep_size
        # We compute the feature size dynamically
        feat_size = self._feature_size(input_channels, convfeat)
        self.fc = nn.Linear(feat_size, rep_size)
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    def _feature_size(self, input_channels, convfeat):
        # Helper to infer size of conv output
        # Pass a dummy tensor through conv layers
        dummy = torch.zeros(1, input_channels, 84, 84)
        x = F.relu(self.conv1(dummy))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.view(1, -1).size(1)

    def forward(self, x):
        """
        Forward pass: x should be [B, C, H, W] float32 in [0,1]
        Returns embeddings [B, rep_size]
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class RNDPredictor(nn.Module):
    """
    Trainable predictor network for RND.  Matches the architecture of RNDTarget but with additional hidden layers.
    """
    def __init__(self, input_channels=4, convfeat=32, rep_size=512, hidden_size=256):
        super(RNDPredictor, self).__init__()
        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(input_channels, convfeat, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(convfeat, convfeat * 2, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(convfeat * 2, convfeat * 2, kernel_size=3, stride=1)
        # Final MLP layers to rep_size
        feat_size = self._feature_size(input_channels, convfeat)
        self.fc1 = nn.Linear(feat_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, rep_size)

    def _feature_size(self, input_channels, convfeat):
        # Same as in target net for consistency
        dummy = torch.zeros(1, input_channels, 84, 84)
        x = F.relu(self.conv1(dummy))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.view(1, -1).size(1)

    def forward(self, x):
        """
        Forward pass: x should be [B, C, H, W] float32 in [0,1]
        Returns embeddings [B, rep_size]
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
