"""Compute depth maps for images in the input folder.
"""
import os
import glob
import utils
import cv2

import tensorflow as tf

from transforms import Resize, NormalizeImage, PrepareForNet

def run(input_path, output_path, model_path):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    print("initialize")

    # the runtime initialization will not allocate all memory on the device to avoid out of GPU memory
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        for gpu in gpus:
          #tf.config.experimental.set_memory_growth(gpu, True)
          tf.config.experimental.set_virtual_device_configuration(gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])
      except RuntimeError as e:
        print(e)

    # load network
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(model_path, 'rb') as f:
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    
    model_operations = tf.compat.v1.get_default_graph().get_operations()
    input_node = '0:0'
    output_layer = model_operations[len(model_operations) - 1].name + ':0'
    print("Last layer name: ", output_layer)

    resize_image = Resize(
                384,
                384,
                resize_target=None,
                keep_aspect_ratio=False,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            )
    
    def compose2(f1, f2):
        return lambda x: f2(f1(x))

    transform = compose2(resize_image, PrepareForNet())

    # get input
    img_names = glob.glob(os.path.join(input_path, "*"))
    num_images = len(img_names)

    # create output folder
    os.makedirs(output_path, exist_ok=True)

    print("start processing")

    with tf.compat.v1.Session() as sess:
      try:
        # load images
        for ind, img_name in enumerate(img_names):

            print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))

            # input
            img = utils.read_image(img_name)
            img_input = transform({"image": img})["image"]

            # compute
            prob_tensor = sess.graph.get_tensor_by_name(output_layer)
            prediction, = sess.run(prob_tensor, {input_node: [img_input] })
            prediction = prediction.reshape(384, 384)
            prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
            
            # output
            filename = os.path.join(
                output_path, os.path.splitext(os.path.basename(img_name))[0]
            )
            utils.write_depth(filename, prediction, bits=2)

      except KeyError:
        print ("Couldn't find input node: ' + input_node + ' or output layer: " + output_layer + ".")
        exit(-1)

    print("finished")


if __name__ == "__main__":
    # set paths
    INPUT_PATH = "input"
    OUTPUT_PATH = "output"
    MODEL_PATH = "model-f46da743.pb"

    # compute depth maps
    run(INPUT_PATH, OUTPUT_PATH, MODEL_PATH)
