# residual_block.py

"""
This script implements a Deep Residual Block for neural network optimization.
It leverages the residual learning hypothesis by using identity shortcut 
connections to reformulate the mapping task, allowing for more efficient 
training of deep architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Initializes a residual block that fits F(x) := H(x) - x.
        
        Inputs:
            in_channels (int): Number of input feature maps.
            out_channels (int): Number of output feature maps.
            stride (int): Stride for the first convolution to control spatial dimensions.
            
        Outputs: 
            An initialized ResidualBlock instance.
        """
        super(ResidualBlock, self).__init__()
        
        # Stacked non-linear layers F(x)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut Connection (Identity or Projection)
        self.shortcut = nn.Identity()
        
        # Dimension Alignment: If dimensions change, use a 1x1 convolution 
        # to project x to the required output dimension.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """
        Performs the forward pass recasting the mapping to H(x) = F(x) + x.
        
        Inputs:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
            
        Outputs:
            torch.Tensor: Output tensor after element-wise addition and ReLU activation.
        """
        # Calculate residual mapping F(x)
        identity = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Core Residual Learning: H(x) = F(x) + x
        # Element-wise addition of the shortcut and the stacked layers
        out += identity
        out = F.relu(out)
        
        return out

# --- Verification Logic ---

if __name__ == "__main__":
    # Example: Processing a 64x64 image with 64 channels
    input_tensor = torch.randn(1, 64, 64, 64)
    res_block = ResidualBlock(64, 64)
    
    output = res_block(input_tensor)
    
    print(f"Input Shape: {input_tensor.shape}")
    print(f"Output Shape: {output.shape}")
    print("Residual mapping successfully applied via identity shortcut.")