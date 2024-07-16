import torch
import torch.nn as nn
from src.models.modules.WavLM import WavLM, WavLMConfig
from src.models.modules.MFA import MFAClassifier

class WavLM_MFA(nn.Module):
    def __init__(self, freeze=False):
        super(WavLM_MFA, self).__init__()

        checkpoint = torch.load('src/models/modules/WavLM-Large.pt')
        cfg = WavLMConfig(checkpoint['cfg'])
        self.cfg = cfg
        self.frontend = WavLM(cfg)
        self.frontend.load_state_dict(checkpoint['model'])
        self.freeze = freeze
        self.num_layers = cfg.encoder_layers
        self.fc = nn.Linear(1024, 2)

        if self.freeze:
            self.frontend.eval()
        
        embed_dim = cfg.encoder_embed_dim
        self.MFA = MFAClassifier(self.num_layers, input_dim=embed_dim, hidden_dim=embed_dim, num_classes=2)

    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                features, layer_results = self.frontend.extract_features(x, output_layer=self.frontend.cfg.encoder_layers, ret_layer_results=True)[0]
        else:
            features, layer_results = self.frontend.extract_features(x, output_layer=self.frontend.cfg.encoder_layers, ret_layer_results=True)[0]
        layer_results = [x.transpose(0, 1) for x, _ in layer_results][:self.num_layers]
        # features = self.frontend.extract_features(x)[0]

        output = self.MFA(layer_results)
        output = self.fc(output)

        return output