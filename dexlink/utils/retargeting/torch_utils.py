import torch.nn as nn


class FCN(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_sizes=(256, 256),
        activation=nn.GELU,
        last_activation=None,
    ):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(activation())
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        if last_activation is not None:
            layers.append(last_activation())
        self.net = nn.Sequential(*layers)

        # Initialize weights to be xavier uniform
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
