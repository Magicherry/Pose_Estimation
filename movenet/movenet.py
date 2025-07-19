import tensorflow as tf
import tensorflow_hub as hub
# from tensorflow_docs.vis import embed
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import imageio
from IPython.display import HTML, display
import time
import pdb


# model_name = "movenet_lightning"
# if "movenet_lightning" in model_name:
#     module = hub.load("singlepose/lightning/4")
#     input_size = 192
# elif "movenet_thunder" in model_name:
#     module = hub.load("singlepose/thunder/4")
#     input_size = 256
# else:
#     raise ValueError("Unsupported model name: %s" % model_name)

# def movenet(input_image):
#     """Runs detection on an input image.

#     Args:
#         input_image: A [1, height, width, 3] tensor represents the input image
#         pixels. Note that the height/width should already be resized and match the
#         expected input resolution of the model before passing into this function.

#     Returns:
#         A [1, 1, 17, 3] float numpy array representing the predicted keypoint
#         coordinates and scores.
#     """
#     model = module.signatures['serving_default']

#     # SavedModel format expects tensor type of int32.
#     input_image = tf.cast(input_image, dtype=tf.int32)
#     # Run model inference.
#     outputs = model(input_image)
#     # Output is a [1, 1, 17, 3] tensor.
#     keypoints_with_scores = outputs['output_0'].numpy()
#     return keypoints_with_scores

class Movenet:
    def __init__(self, model_name):
        if "movenet_lightning" in model_name:
            self.module = hub.load("movenet/singlepose/lightning/4")
            self.input_size = 192
        elif "movenet_thunder" in model_name:
            self.module = hub.load("movenet/singlepose/thunder/4")
            self.input_size = 256
        else:
            raise ValueError("Unsupported model name: %s" % model_name)
        self.model = self.module.signatures['serving_default']

    def predict(self, image, verbose=False):
        t1 = time.time()
        image_h, image_w, _ = image.shape
        input_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)
        input_image = tf.expand_dims(input_image, axis=0)
        input_image = tf.image.resize_with_pad(input_image, self.input_size, self.input_size)
        input_image = tf.cast(input_image, dtype=tf.int32)
        outputs = self.model(input_image)
        keypoints_with_scores = outputs['output_0'].numpy()
        keypoints_with_scores = keypoints_with_scores[0][0]
        offset = int(abs(image_h - image_w) / 2)
        offset_pixel = [0, offset] if image_h > image_w else [offset, 0]
        offset_pixel = np.array(offset_pixel, dtype=np.int32)
        keypoints_with_scores[:, :2] *= max(image_h, image_w)
        keypoints_with_scores[:, :2] -= offset_pixel
        keypoints_with_scores = keypoints_with_scores[:, [1, 0, 2]]
        t2 = time.time()
        if verbose:
            print("time use {}ms".format(int((t2 - t1) * 1000)))
        return keypoints_with_scores


if __name__ == '__main__':
    image_path = '1.jpg'
    model = Movenet("movenet_lightning")
    image = cv2.imread(image_path)
    for i in range(10):
        keypoints_with_scores = model.predict(image, verbose=True)

    # # Load the input image.
    # image_path = '1.jpg'
    # image = tf.io.read_file(image_path)
    # image = tf.image.decode_jpeg(image)  # RGB
    # image_h, image_w, _ = image.shape

    # # Resize and pad the image to keep the aspect ratio and fit the expected size.
    # input_image = tf.expand_dims(image, axis=0)
    # input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

    # # Run model inference.
    # t1 = time.time()
    # keypoints_with_scores = movenet(input_image)  # [batch, num, 17, 3]   3 = y, x, score
    # keypoints_with_scores = keypoints_with_scores[0][0]
    # t2 = time.time()
    # print("frame use time {}ms".format(int((t2-t1)*1000)))

    # # 点坐标还原到图像尺寸，乘比例（input_size）减去offset
    # ratio = max(image_h, image_w) / input_size
    # offset = int(abs(image_h - image_w) / 2)
    # offset_pixel = [0, offset] if image_h > image_w else [offset, 0]
    # offset_pixel = np.array(offset_pixel, dtype=np.int32)
    # keypoints_with_scores[:, :2] *= max(image_h, image_w)
    # keypoints_with_scores[:, :2] -= offset_pixel
    # # pdb.set_trace()

    # display_image = cv2.cvtColor(np.array(image, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    # for kp in keypoints_with_scores:
    #     y, x, score = int(kp[0]), int(kp[1]), kp[2]
    #     cv2.circle(display_image, [x, y], 2, (255, 0, 0), 2)
    # res_path = image_path.replace('.', '_res.')
    # cv2.imwrite(res_path, display_image)
