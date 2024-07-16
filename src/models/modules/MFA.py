import torch
import torch.nn as nn
from src.models.modules.ASP import ASP

class MFAClassifier(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, num_classes):
        super(MFAClassifier, self).__init__()
        self.t_asp = nn.ModuleList([ASP(input_dim) for _ in range(num_layers)])
        self.l_asp = ASP(input_dim)  # Each T-ASP outputs a vector of size input_dim * 2
        
        self.lin_proj1 = nn.Linear(input_dim * 2, input_dim)
        self.lin_proj2 = nn.Linear(input_dim * 2, input_dim)
        
        self.batch_norm = nn.BatchNorm1d(input_dim)

    def forward(self, x):
        # h shape: [batch_size, num_layers, time_steps, feature_dim]
        x = torch.stack(x, dim=1)
        t_asp_outputs = []
        for i in range(len(self.t_asp)):
            t_asp_output = self.t_asp[i](x[:, i, :, :])
            t_asp_outputs.append(t_asp_output)

        t_asp_outputs = torch.stack(t_asp_outputs, dim=1)  # shape: [batch_size, num_layers, hidden_dim]
        t_asp_outputs = self.lin_proj1(t_asp_outputs)

        l_asp_output = self.l_asp(t_asp_outputs)  # shape: [batch_size, hidden_dim]
        l_asp_output = self.lin_proj2(l_asp_output)
        l_asp_output = self.batch_norm(l_asp_output)

        return l_asp_output