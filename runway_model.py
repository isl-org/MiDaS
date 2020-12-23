import runway
import cv2
import torch
import numpy as np

@runway.setup(options={'checkpoint_dir': runway.file(extension='.pt', description="checkpoint file"),})
def setup(opts):

    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    # midas = torch.load(opts['checkpoint_dir'])
    return midas


@runway.command('translate', inputs={'source_imgs': runway.image(description='input image to be translated'),}, outputs={'image': runway.image(description='output image containing the translated result')})
def translate(midas, inputs):
    cv_image = np.array(inputs['source_imgs']) 
    img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform
    input_batch = transform(img)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    output = prediction.numpy()
   
    return output
    
if __name__ == '__main__':
    runway.run(port=8889)
