import cv2
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from src.models.feature.ResNet import ResNet18
from example.cam.visualization import CAM

IMG_PATH = './test_img/stl10/test2.png'
MODEL_PATH = './pretrain/resnet18_stl10_gap_model.pth'

if torch.cuda.is_available():

    device = 'cuda'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = 'cpu'
    torch.set_default_tensor_type('torch.FloatTensor')

class_name = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']


def drawing(cam, img_path, shape=(128, 128)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, shape)

    fig, axs = plt.subplots(1, 3, figsize=(10, 10))

    axs[0].imshow(cam)

    resized_cam = cv2.resize(cam, shape)

    axs[1].imshow(resized_cam)

    heatmap = cv2.applyColorMap(np.uint8(255 * resized_cam), cv2.COLORMAP_JET)

    heatimg = heatmap * 0.3 + img * 0.5

    print(heatimg.shape)

    cv2.imwrite('./cam.jpg', heatimg)

    cam_img = cv2.imread('./cam.jpg')
    cam_img = cv2.cvtColor(heatimg, cv2.COLOR_BGR2RGB)

    axs[2].imshow(cam_img)


model = ResNet18(classes=len(class_name),alpha=0).to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model = model.eval()

img = Image.open(IMG_PATH)

transformer = transforms.Compose([transforms.Resize((128, 128)),
                                  transforms.ToTensor()])

tensor_img = transformer(img).to(device)
tensor_img = tensor_img.view(1, 3, 128, 128)

# print(model._modules)
final_conv_name = 'Block4'

# hook the feature extractor
features_blobs = []

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

model._modules.get(final_conv_name).register_forward_hook(hook_feature)

# CAM
cam = CAM()
cam_img = cam.get_cam(model, features_blobs, tensor_img)

drawing(IMG_PATH, cam_img)