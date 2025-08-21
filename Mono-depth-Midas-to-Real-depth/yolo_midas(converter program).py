import os
import glob
import time
import argparse
import numpy as np
import torch
import cv2
from imutils.video import VideoStream
from midas.model_loader import default_models, load_model
import utils
from ultralytics import YOLO

first_execution = True

# set base depth (input real depth in bottom right yello box)
REF_REAL_M = 15.0

def process(device, model, model_type, image, input_size, target_size, optimize, use_camera):
    global first_execution
    sample = torch.from_numpy(image).to(device).unsqueeze(0)

    # convert image to Torch Tensor form numpy and send GPU/CPU
    if optimize and device == torch.device("cuda"):
        if first_execution:
            print("  Optimization to half-floats activated...")
        sample = sample.to(memory_format=torch.channels_last)
        sample = sample.half()

    if first_execution or not use_camera:
        height, width = sample.shape[2:]
        print(f"    Input resized to {width}x{height} before entering the encoder")
        first_execution = False
    # image scaling with bicubic
    prediction = model.forward(sample)
    prediction = (
        torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=target_size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        .squeeze()
        .cpu()
        .numpy()
    )
    return prediction

def create_side_by_side(image, depth, grayscale):
    depth_min = depth.min()
    depth_max = depth.max()
    normalized_depth = 255 * (depth - depth_min) / (depth_max - depth_min)
    normalized_depth *= 3

    right_side = np.repeat(np.expand_dims(normalized_depth, 2), 3, axis=2) / 3
    if not grayscale:
        right_side = cv2.applyColorMap(np.uint8(right_side), cv2.COLORMAP_INFERNO)

    if image is None:
        return right_side
    else:
        return np.concatenate((image, right_side), axis=1)

def run(input_path, output_path, model_path, model_type="dpt_swin2_large_384", optimize=False, height=None):
    print("Initialize")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize, height)
    yolo_model = YOLO("yolo11n.pt")

    print("Start processing")

    with torch.no_grad():
        fps = 1
        video = VideoStream(0).start()
        time_start = time.time()
        frame_index = 0
        while True:
            frame = video.read()
            if frame is not None:
                # YOLO object detection
                yolo_results = yolo_model(frame)[0]
                yolo_annotated = frame.copy()

                # MiDaS depth estimation
                original_image_rgb = np.flip(frame, 2)
                image = transform({"image": original_image_rgb / 255})["image"]
                prediction = process(device, model, model_type, image, (net_w, net_h), original_image_rgb.shape[1::-1], optimize, True)

                # set base box (bottom right yello box)
                h, w = prediction.shape
                box_w, box_h = 50, 50
                box_x, box_y = w - box_w - 10, h - box_h - 10
                ref_depth = np.mean(prediction[box_y:box_y+box_h, box_x:box_x+box_w])
                cv2.rectangle(yolo_annotated, (box_x, box_y), (box_x+box_w, box_y+box_h), (0, 255, 255), 2)

                # real depth calculation function
                def calculate_distance(depth_value, ref_depth):
                    if ref_depth == 0:
                        return float('inf')
                    return REF_REAL_M * (ref_depth / (depth_value + 1e-6))

                # visualize object detection
                for i, box in enumerate(yolo_results.boxes.xyxy):
                    x1, y1, x2, y2 = map(int, box[:4])
                    cls_id = int(yolo_results.boxes.cls[i])
                    label = yolo_model.names[cls_id]

                    depth_crop = prediction[y1:y2, x1:x2]
                    if depth_crop.size == 0:
                        continue

                    avg_depth = np.mean(depth_crop)
                    estimated_distance = calculate_distance(avg_depth, ref_depth)

                    text = f"{label} {estimated_distance:.2f}m"
                    color = (0, 255, 0)

                    cv2.rectangle(yolo_annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(yolo_annotated, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # combin YOLO and MiDaS
                combined = np.hstack((yolo_annotated, create_side_by_side(None, prediction, True)))
                cv2.putText(combined, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # output
                cv2.imshow("YOLO + MiDaS + Reference Box", combined / 255)
                if cv2.waitKey(1) == 27:
                    break

                frame_index += 1
                fps = (1 - 0.1) * fps + 0.1 * 1 / (time.time() - time_start)
                time_start = time.time()

        print("Finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', default=None)
    parser.add_argument('-o', '--output_path', default=None)
    parser.add_argument('-m', '--model_weights', default=None)
    parser.add_argument('-t', '--model_type', default='dpt_swin2_large_384')
    parser.add_argument('--optimize', dest='optimize', action='store_true')
    parser.add_argument('--height', type=int, default=None)
    args = parser.parse_args()

    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    run(args.input_path, args.output_path, args.model_weights, args.model_type, args.optimize, height=args.height)
