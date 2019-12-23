import numpy as np
from torch.nn import functional as F


def min_max_scaling(matrix):
    matrix = (matrix - np.min(matrix)) / np.max(matrix) - np.min(matrix)
    return matrix


def standard_scaling(matrix):
    matrix = (matrix - np.mean(matrix)) / np.std(matrix)
    return matrix


def scaling(matrix):
    matrix = matrix - np.min(matrix)
    matrix = matrix / np.max(matrix)

    return matrix


class CAM(object):
    def __init__(self):
        self.feature_blobs = []

    def hook_feature(self, module, input, output):
        self.feature_blobs.append(output.cpu().data.numpy())

    def get_cam(self, model, features_blobs, tensor_img):
        # weights before softmax
        params = list(model.parameters())
        class_weights = np.squeeze(params[-2].cpu().data.numpy())

        output = model(tensor_img)
        output = F.softmax(output, dim=1).data.squeeze()
        pred = output.argmax(0).item()

        final_conv = features_blobs[0][0]

        cam = np.zeros(dtype=np.float32, shape=final_conv.shape[1:3])

        for i, w in enumerate(class_weights[pred]):
            cam += w * final_conv[i, :, :]

        cam = scaling(cam)

        return cam