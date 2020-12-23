import runway
import torch
import cv2
import torch

@runway.setup(options={'checkpoint_dir': runway.file(description="runs folder"),})
def setup(opts):
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    midas.eval()
    use_large_model = True

    if use_large_model:
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    else:
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if use_large_model:
        transform = midas_transforms.default_transform
    else:
        transform = midas_transforms.small_transform
    return model


@runway.command('translate', inputs={'source_imgs': runway.image(description='input image to be translated'),}, outputs={'image': runway.image(description='output image containing the translated result')})
def translate(model, inputs):
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
