import torch
import torch.nn as nn
import torchvision.models as models

class PoseNet(nn.Module):
    """
    A lightweight CNN for estimating the periods (t_x, t_y) and intra-tile
    offsets of a repeating grid.
    """
    def __init__(self, backbone='mobilenet_v2', pretrained=True, t_min=(4.0, 4.0), t_max=(256.0, 256.0), tile_dims=(5.0, 3.0)):
        super(PoseNet, self).__init__()

        if backbone == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
            
            # Modify the first conv layer for 1-channel (grayscale) images
            original_conv1_weights = self.backbone.features[0][0].weight.data
            self.backbone.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            self.backbone.features[0][0].weight.data = torch.mean(original_conv1_weights, dim=1, keepdim=True)

            num_features = self.backbone.last_channel
            self.backbone.classifier = nn.Identity()
        
        elif backbone == 'mobilenet_v3_small':
            self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
            
            # Modify the first conv layer for 1-channel (grayscale) images
            original_conv1_weights = self.backbone.features[0][0].weight.data
            self.backbone.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
            self.backbone.features[0][0].weight.data = torch.mean(original_conv1_weights, dim=1, keepdim=True)

            num_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise NotImplementedError("Backbone not supported")

        self.head = nn.Linear(num_features, 4) # 2 for periods, 2 for offsets
        
        self.register_buffer('t_min', torch.tensor(t_min, dtype=torch.float32))
        self.register_buffer('t_max', torch.tensor(t_max, dtype=torch.float32))
        self.register_buffer('tile_dims', torch.tensor(tile_dims, dtype=torch.float32))

    def forward(self, image):
        """
        The forward pass returns the predicted periods (t_x, t_y) and offsets.
        
        Args:
            image (torch.Tensor): The input image tensor.
        
        Returns:
            A tuple containing:
            - periods (torch.Tensor): The predicted periods (t_x_px, t_y_px).
            - offsets (torch.Tensor): The predicted intra-tile offsets (tx_OCS, ty_OCS).
        """
        features = self.backbone(image)
        u = self.head(features) # (B, 4)
        
        # Unpack predictions
        u_periods = u[:, :2]
        u_offsets = u[:, 2:]

        # --- Period Prediction ---
        r_periods = torch.sigmoid(u_periods) # (B, 2), range (0, 1)
        periods = self.t_min + (self.t_max - self.t_min) * r_periods
        
        # --- Offset Prediction ---
        # Sigmoid maps to (0, 1), which we scale by the tile dimensions
        r_offsets = torch.sigmoid(u_offsets)
        offsets = r_offsets * self.tile_dims

        return periods, offsets
