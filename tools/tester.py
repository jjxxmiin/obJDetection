import numpy as np
import cv2


def save_tensor_image(image, boxes=None, saved_path='test.png'):
    '''
    :param image: (tensor) cpu image
    :return: (file) save image
    '''

    image = image.permute(1, 2, 0).numpy() * 255.0
    image = image.astype('uint8')

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if boxes is not None:
        for box in boxes:
            loc = np.floor(box)
            image = cv2.rectangle(image, (loc[0], loc[1]), (loc[2], loc[3]), (255, 0, 0), 3)

    cv2.imwrite(saved_path, image)

    print('Finish image save testing')
