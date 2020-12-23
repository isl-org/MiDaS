import runway
import cv2
import torch

@runway.setup(options={'checkpoint_dir': runway.file(extension='.pt', description="checkpoint file"),})
def setup(opts):

 
    midas = torch.load(opts['checkpoint_dir'])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform
    return transform, midas


@runway.command('translate', inputs={'source_imgs': runway.image(description='input image to be translated'),}, outputs={'image': runway.image(description='output image containing the translated result')})
def translate(transform,midas, inputs):
    img = cv2.imread(inputs['source_imgs'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    output = prediction.cpu().numpy()
   
    return output




if __name__ == '__main__':
    runway.run(port=8889)
