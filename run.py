"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
from monodepth_net import MonoDepthNet
import utils


def run(input_path, output_path, model_path):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    print("initialize")

    # select device
    device = torch.device("cpu")
    print("device: %s" % device)

    # load network
    model = MonoDepthNet(model_path)
    model.to(device)
    model.eval()

    # get input
    img_names = glob.glob(os.path.join(input_path, "*"))
    num_images = len(img_names)

    # create output folder
    os.makedirs(output_path, exist_ok=True)

    print("start processing")

    for ind, img_name in enumerate(img_names):

        print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))

        # input
        img = utils.read_image(img_name)

        img_input = utils.resize_image(img)
        img_input = img_input.to(device)

        # compute
        with torch.no_grad():
            out = model.forward(img_input)

        depth = utils.resize_depth(out, img.shape[1], img.shape[0])

        # output
        filename = os.path.join(
            output_path, os.path.splitext(os.path.basename(img_name))[0]
        )
        utils.write_depth(filename, depth, bits=2)

    print("finished")


if __name__ == "__main__":
    # set paths
    INPUT_PATH = "input"
    OUTPUT_PATH = "output"
    MODEL_PATH = "model.pt"

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    run(INPUT_PATH, OUTPUT_PATH, MODEL_PATH)
