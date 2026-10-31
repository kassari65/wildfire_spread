import argparse
import torch
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.models.UTAELightning import UTAELightning
from src.models.utae_paps_models.utae import UTAE


def load_utae_model(weights_path: str, device: str = "cpu", n_channels: int = 40):
    """
    Load trained UTAE weights from either a PyTorch Lightning .ckpt file
    or a raw state_dict (.pth / .pth.tar / .pt).
    """
    print(f"Loading weights from: {weights_path}")
    raw_state = torch.load(weights_path, map_location=device)

    # Extract state_dict
    if isinstance(raw_state, dict) and "state_dict" in raw_state:
        state_dict = raw_state["state_dict"]
    else:
        state_dict = raw_state

    # Clean prefixes like "model." or "module."
    clean_dict = {}
    for k, v in state_dict.items():
        clean_k = k
        if clean_k.startswith("model."):
            clean_k = clean_k[6:]
        elif clean_k.startswith("module."):
            clean_k = clean_k[7:]
        clean_dict[clean_k] = v

    # Instantiate pure UTAE architecture
    model = UTAE(
        input_dim=n_channels,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        out_conv=[32, 1],
        str_conv_k=4,
        str_conv_s=2,
        str_conv_p=1,
        agg_mode="att_group",
        encoder_norm="group",
        n_head=16,
        d_model=256,
        d_k=4,
        encoder=False,
        return_maps=False,
        pad_value=0,
        padding_mode="reflect",
    )

    # Filter keys matching the model
    model_state = model.state_dict()
    matched_dict = {
        k: v for k, v in clean_dict.items()
        if k in model_state and model_state[k].shape == v.shape
    }

    model.load_state_dict(matched_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"Successfully loaded {len(matched_dict)} / {len(model_state)} layers into UTAE.")
    return model


def main():
    parser = argparse.ArgumentParser(description="Test loading trained UTAE model weights and run dummy inference.")
    parser.add_argument("--weights_path", required=True, help="Path to .ckpt or .pth.tar weights file.")
    parser.add_argument("--device", default="cpu", help="Device to load model on ('cpu' or 'cuda').")
    parser.add_argument("--n_channels", type=int, default=40, help="Number of input feature channels.")
    parser.add_argument("--T", type=int, default=5, help="Number of temporal observation steps.")
    args = parser.parse_args()

    model = load_utae_model(args.weights_path, device=args.device, n_channels=args.n_channels)

    # Dummy inference test [B, T, C, H, W]
    dummy_x = torch.randn(1, args.T, args.n_channels, 64, 64, device=args.device)
    dummy_doys = torch.tensor([[100, 101, 102, 103, 104]], device=args.device)

    print("\nRunning test forward pass...")
    with torch.no_grad():
        logits = model(dummy_x, batch_positions=dummy_doys, return_att=False)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

    print(f"Logits shape : {logits.shape}")
    print(f"Probs shape  : {probs.shape} (min: {probs.min().item():.4f}, max: {probs.max().item():.4f})")
    print(f"Preds shape  : {preds.shape}")
    print("✅ Model inference verification completed successfully!")


if __name__ == "__main__":
    main()
