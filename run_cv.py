import cv2
"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import utils
import cv2

from torchvision.transforms import Compose
from models.midas_net import MidasNet
from models.transforms import Resize, NormalizeImage, PrepareForNet


def run(model_path):
    """Run MonoDepthNN to compute depth maps.

    Args:
        model_path (str): path to saved model
    """
    print("initialize")

    # select device
    device = torch.device("cuda")
    print("device: %s" % device)

    # load network
    model = MidasNet(model_path, non_negative=True)

    transform = Compose(
        [
            Resize(
                384,
                384,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(1)

    print("start processing")

    while cap.isOpened():

        ret, frame = cap.read()

        if ret:
            img_input = transform({"image": frame})["image"]

            # compute
            with torch.no_grad():
                sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
                prediction = model.forward(sample)
                prediction = (
                    torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=frame.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    )
                    .squeeze()
                    .cpu()
                    .numpy()
                )

            # output

            # utils.write_depth(filename, prediction, bits=2)

            cv2.imshow('frame', frame)
            cv2.imshow('prediction', prediction)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Camera is not recording")

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    print("finished")


if __name__ == "__main__":
    MODEL_PATH = "model.pt"

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    run(MODEL_PATH)
