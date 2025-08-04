import argparse
import numpy as np
import requests
import torch
import cv2

from midas.model_loader import default_models, load_model
from run import process


def stream_depth(server_url, camera_source=0, model_type="dpt_beit_large_512", model_weights=None, optimize=False):
    """Stream depth predictions via HTTP POST."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_weights is None:
        model_weights = default_models[model_type]

    model, transform, net_w, net_h = load_model(device, model_weights, model_type, optimize)

    src = int(camera_source) if str(camera_source).isdigit() else camera_source
    cap = cv2.VideoCapture(src)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = transform({"image": image_rgb / 255.0})["image"]

            with torch.no_grad():
                prediction = process(
                    device,
                    model,
                    model_type,
                    image,
                    (net_w, net_h),
                    image_rgb.shape[1::-1],
                    optimize,
                    True,
                )

            requests.post(
                server_url,
                data=prediction.astype(np.float32).tobytes(),
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Width": str(prediction.shape[1]),
                    "X-Height": str(prediction.shape[0]),
                },
                timeout=1,
            )
    finally:
        cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream depth maps to another process.")
    parser.add_argument("--server_url", required=True, help="URL to POST depth arrays to")
    parser.add_argument("--camera_source", default=0, help="Camera device index or video/stream URL")
    parser.add_argument("--model_type", default="dpt_beit_large_512", help="Model type")
    parser.add_argument("--model_weights", default=None, help="Path to model weights")
    parser.add_argument("--optimize", action="store_true", help="Use half-float optimization")
    args = parser.parse_args()

    stream_depth(
        args.server_url,
        args.camera_source,
        args.model_type,
        args.model_weights,
        args.optimize,
    )

