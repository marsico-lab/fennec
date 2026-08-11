from torch import nn

class TCNResidualBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, dropout_p=0.3):
        super(TCNResidualBlock1d, self).__init__()
        # Compute left padding so that output length = input length
        pad_size = (kernel_size - 1) * dilation

        # First causal convolution
        self.conv1 = nn.Sequential(
            nn.ConstantPad1d((pad_size, 0), 0),  # Only pad on left
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, dilation=dilation, padding=0),
            nn.BatchNorm1d(out_channels),
            nn.ELU(),
            nn.Dropout(p=dropout_p)
        )
        # Second causal convolution
        self.conv2 = nn.Sequential(
            nn.ConstantPad1d((pad_size, 0), 0),
            nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, dilation=dilation, padding=0),
            nn.BatchNorm1d(out_channels),
            nn.ELU(),
            nn.Dropout(p=dropout_p)
        )
        
        # Shortcut connection for residual
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(out_channels)
            )
        self.elu = nn.ELU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        residual = self.shortcut(x)
        return self.elu(out + residual)
