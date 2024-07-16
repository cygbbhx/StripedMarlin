from typing import Dict

from src.models import lcnn, rawnet3, specrnet, HuBERT, WavLM_MFA


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
    else:
        raise ValueError(f"Model '{model_name}' not supported")
