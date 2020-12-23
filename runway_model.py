import runway
import cv2
import torch
import numpy as np
import download_checkpoint


@runway.command('translate', inputs={'source_imgs': runway.image(description='input image to be translated'),'large': boolean(default=True, description='use large model'),}, outputs={'image': runway.image(description='output image containing the translated result')})
def translate(midas, inputs):
    cv_image = np.array(inputs['source_imgs']) 
    img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    if inputs['large']:
        transform = download_checkpoint.transforml
        input_batch = transform(img)
        midas = download_checkpoint.midasl
    else:     
        transform = download_checkpoint.transformS
        input_batch = transform(img)
        midas = download_checkpoint.midasS

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
