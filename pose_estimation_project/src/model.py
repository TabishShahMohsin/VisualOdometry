import torch
import torch.nn as nn
import torchvision.models as models

class PeriodNet(nn.Module):
    """
    A lightweight CNN for estimating the periods (t_x, t_y) of a repeating grid.
    """
    def __init__(self, backbone='mobilenet_v3_small', pretrained=True, t_min=(4.0, 4.0), t_max=(256.0, 256.0)):
        super(PeriodNet, self).__init__()

        if backbone == 'mobilenet_v3_small':
            self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
            
            # Modify the first conv layer for 1-channel (grayscale) images
            original_conv1_weights = self.backbone.features[0][0].weight.data
            self.backbone.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
            self.backbone.features[0][0].weight.data = torch.mean(original_conv1_weights, dim=1, keepdim=True)

            num_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise NotImplementedError("Backbone not supported")

        self.head = nn.Linear(num_features, 2)
        
        self.register_buffer('t_min', torch.tensor(t_min, dtype=torch.float32))
        self.register_buffer('t_max', torch.tensor(t_max, dtype=torch.float32))

    def forward(self, image):
        """
        The forward pass returns the predicted periods (t_x, t_y).
        
        Args:
            image (torch.Tensor): The input image tensor.
        """
        features = self.backbone(image)
        u = self.head(features) # (B, 2)
        r = torch.sigmoid(u)     # (B, 2), range (0, 1)
        
        # Map to the desired period range
        periods = self.t_min + (self.t_max - self.t_min) * r
        
        return periods