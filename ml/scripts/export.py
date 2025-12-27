"""
Model Export Script

Exports trained PyTorch model to ONNX format for production deployment.

Usage:
    python export.py --checkpoint checkpoints/best_model.pth --output models/model.onnx
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import yaml
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import create_model


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_size: tuple = (1, 3, 224, 224),
    opset_version: int = 13,
    dynamic_batch: bool = True,
    verify: bool = True
) -> None:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model
        output_path: Path to save ONNX model
        input_size: Input tensor size (batch, channels, height, width)
        opset_version: ONNX opset version
        dynamic_batch: Whether to allow dynamic batch size
        verify: Whether to verify the exported model
    """
    model.eval()
    
    # Create dummy input
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size, device=device)
    
    # Define dynamic axes if needed
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    
    # Export
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes
    )
    
    logger.info(f"Model exported to {output_path}")
    
    # Verify
    if verify:
        try:
            import onnx
            import onnxruntime as ort
            
            # Check model
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model structure verified!")
            
            # Test inference
            ort_session = ort.InferenceSession(
                str(output_path),
                providers=['CPUExecutionProvider']
            )
            
            # Run inference
            input_name = ort_session.get_inputs()[0].name
            ort_inputs = {input_name: dummy_input.cpu().numpy()}
            ort_outputs = ort_session.run(None, ort_inputs)
            
            # Compare with PyTorch output
            with torch.no_grad():
                torch_output = model(dummy_input).cpu().numpy()
            
            # Check if outputs are close
            import numpy as np
            if np.allclose(torch_output, ort_outputs[0], rtol=1e-3, atol=1e-5):
                logger.info("ONNX model output verified - matches PyTorch output!")
            else:
                logger.warning("ONNX output differs from PyTorch output!")
                max_diff = np.max(np.abs(torch_output - ort_outputs[0]))
                logger.warning(f"Maximum difference: {max_diff}")
                
        except ImportError:
            logger.warning("ONNX or ONNX Runtime not installed, skipping verification")


def export_to_torchscript(
    model: torch.nn.Module,
    output_path: str,
    input_size: tuple = (1, 3, 224, 224),
    method: str = "trace"
) -> None:
    """
    Export PyTorch model to TorchScript format.
    
    Args:
        model: PyTorch model
        output_path: Path to save TorchScript model
        input_size: Input tensor size
        method: Export method ("trace" or "script")
    """
    model.eval()
    
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size, device=device)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if method == "trace":
        traced_model = torch.jit.trace(model, dummy_input)
    else:
        traced_model = torch.jit.script(model)
    
    traced_model.save(str(output_path))
    logger.info(f"TorchScript model exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export Model to ONNX/TorchScript")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/efficientnet_config.yaml",
        help="Path to model configuration"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/model.onnx",
        help="Output path for exported model"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["onnx", "torchscript", "both"],
        default="onnx",
        help="Export format"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version"
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=4,
        default=[1, 3, 224, 224],
        help="Input size (batch channels height width)"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification"
    )
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    
    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    logger.info("Model loaded successfully!")
    
    # Get input size from config or args
    input_size = tuple(args.input_size)
    img_size = config.get("data", {}).get("img_size", 224)
    if input_size[2:] != (img_size, img_size):
        input_size = (input_size[0], input_size[1], img_size, img_size)
    
    logger.info(f"Input size: {input_size}")
    
    # Export
    output_path = Path(args.output)
    
    if args.format in ["onnx", "both"]:
        onnx_path = output_path.with_suffix(".onnx")
        logger.info(f"Exporting to ONNX: {onnx_path}")
        export_to_onnx(
            model,
            str(onnx_path),
            input_size=input_size,
            opset_version=args.opset,
            verify=not args.no_verify
        )
    
    if args.format in ["torchscript", "both"]:
        ts_path = output_path.with_suffix(".pt")
        logger.info(f"Exporting to TorchScript: {ts_path}")
        export_to_torchscript(
            model,
            str(ts_path),
            input_size=input_size
        )
    
    # Print model info
    logger.info("\n" + "="*50)
    logger.info("EXPORT SUMMARY")
    logger.info("="*50)
    
    if args.format in ["onnx", "both"]:
        onnx_path = output_path.with_suffix(".onnx")
        if onnx_path.exists():
            size_mb = onnx_path.stat().st_size / (1024 * 1024)
            logger.info(f"ONNX model: {onnx_path} ({size_mb:.2f} MB)")
    
    if args.format in ["torchscript", "both"]:
        ts_path = output_path.with_suffix(".pt")
        if ts_path.exists():
            size_mb = ts_path.stat().st_size / (1024 * 1024)
            logger.info(f"TorchScript model: {ts_path} ({size_mb:.2f} MB)")
    
    logger.info("="*50)
    logger.info("Export completed successfully!")


if __name__ == "__main__":
    main()
