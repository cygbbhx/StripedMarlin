from typing import Dict

from src.models import lcnn, rawnet3, specrnet, HuBERT, WavLM_MFA, AASIST


def get_model(model_name: str, config: Dict, device: str):
    if model_name == "rawnet3":
        return rawnet3.prepare_model()
    elif model_name == "lcnn":
        return lcnn.LCNN(device=device, **config)
    elif model_name == "specrnet":
        return specrnet.SpecRNet(
            specrnet.get_config(config.get("input_channels", 1)),
            device=device,
            **config,
        )
    elif model_name == "HuBERT":
        return HuBERT.get_model()
    elif model_name == "WavLM":
        return WavLM_MFA.WavLM_MFA(freeze=False)
    elif model_name == "AASIST":
        model_config = {
        #"model_path": "/home/work/StripedMarlin/jimin/StripedMarlin/src/models/weights/AASIST-L.pth",
        "freeze": False, 
        "architecture": "AASIST",
        "nb_samp": 64600,
        "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
        "gat_dims": [24, 32],
        "pool_ratios": [0.4, 0.5, 0.7, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0]
        }
        return AASIST.get_model(model_config, device)
    else:
        raise ValueError(f"Model '{model_name}' not supported")
