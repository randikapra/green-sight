"""
Model factory: builds models from config.
Handles 4-channel input adaptation and layer freezing.
"""

import torch
import torch.nn as nn
import timm
from typing import Optional


class AdaptiveInputLayer(nn.Module):
    """1x1 conv to adapt N-channel input to 3-channel for pretrained backbones."""
    def __init__(self, in_channels: int = 4, out_channels: int = 3):
        super().__init__()
        self.adapt = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # Init: roughly preserve channel information
        nn.init.kaiming_normal_(self.adapt.weight)
    
    def forward(self, x):
        return self.adapt(x)


def _adapt_first_conv(module: nn.Module, attr: str, in_channels: int):
    """
    Replace a conv layer to accept `in_channels` instead of 3.
    Copies pretrained weights for first 3 channels, averages for extra channels.
    """
    original = getattr(module, attr)
    new_conv = nn.Conv2d(
        in_channels, original.out_channels,
        kernel_size=original.kernel_size,
        stride=original.stride,
        padding=original.padding,
        bias=original.bias is not None,
    )
    with torch.no_grad():
        # Copy RGB weights
        new_conv.weight[:, :3] = original.weight[:, :3]
        # Extra channels: average of RGB weights
        for c in range(3, in_channels):
            new_conv.weight[:, c] = original.weight[:, :3].mean(dim=1)
    setattr(module, attr, new_conv)


def _freeze_layers(model: nn.Module, layer_names: list):
    """Freeze parameters in named layers."""
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = False
                break


def build_model(model_name: str, model_cfg: dict, num_classes: int, 
                input_channels: int = 4) -> nn.Module:
    """
    Build a model from config.
    
    Args:
        model_name: identifier string
        model_cfg: dict from config (backbone, pretrained, input_adapt, etc.)
        num_classes: number of output classes
        input_channels: number of input channels (default 4 for satellite)
    
    Returns:
        nn.Module ready for training
    """
    backbone_name = model_cfg['backbone']
    pretrained = model_cfg.get('pretrained', True)
    input_adapt = model_cfg.get('input_adapt', 'adapter')
    img_size = model_cfg.get('img_size', None)
    freeze = model_cfg.get('freeze_layers', [])
    
    # Build backbone
    create_kwargs = {
        'pretrained': pretrained,
        'num_classes': num_classes,
    }
    if img_size is not None:
        create_kwargs['img_size'] = img_size
    
    # For adapter-based models, create with 3 channels
    if input_adapt == 'adapter':
        create_kwargs['in_chans'] = 3
        backbone = timm.create_model(backbone_name, **create_kwargs)
        model = nn.Sequential(
            AdaptiveInputLayer(input_channels, 3),
            backbone
        )
    else:
        backbone = timm.create_model(backbone_name, **create_kwargs)
        # Directly modify the first conv layer
        if hasattr(backbone, input_adapt):
            _adapt_first_conv(backbone, input_adapt, input_channels)
        else:
            # Try nested access (e.g., "stem.0" for ConvNeXt)
            parts = input_adapt.split('.')
            obj = backbone
            for p in parts[:-1]:
                obj = getattr(obj, p) if not p.isdigit() else obj[int(p)]
            last = parts[-1]
            if last.isdigit():
                # Index into Sequential — replace in-place
                idx = int(last)
                original = obj[idx]
                new_conv = nn.Conv2d(
                    input_channels, original.out_channels,
                    kernel_size=original.kernel_size,
                    stride=original.stride,
                    padding=original.padding,
                    bias=original.bias is not None,
                )
                with torch.no_grad():
                    new_conv.weight[:, :3] = original.weight[:, :3]
                    for c in range(3, input_channels):
                        new_conv.weight[:, c] = original.weight[:, :3].mean(dim=1)
                obj[idx] = new_conv
            else:
                _adapt_first_conv(obj, last, input_channels)
        model = backbone
    
    # Freeze layers
    if freeze:
        _freeze_layers(model, freeze)
    
    return model


def get_model_info(model: nn.Module) -> dict:
    """Get model parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        'total_params': total,
        'trainable_params': trainable,
        'frozen_params': frozen,
        'total_params_M': f"{total/1e6:.1f}M",
        'trainable_params_M': f"{trainable/1e6:.1f}M",
    }
