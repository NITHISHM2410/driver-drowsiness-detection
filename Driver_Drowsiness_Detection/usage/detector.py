import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle


class Drowsiness_Detector:
    def __init__(self, list_models):
        """
        :param list_models: a list containing the path of two models

        """
        self.detector = MTCNN()
        self.models = self.get_models_ready()
        self.list_models = list_models

    def get_models_ready(self):
        eye_model = tf.keras.models.load_model(self.list_models[0])
        yawn_model = tf.keras.models.load_model(self.list_models[1])
        return eye_model, yawn_model

    @staticmethod
    def sharpen(img):
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
        return img

    @staticmethod
    def return_boxes(result_list):
        print(result_list)
        for result in result_list:
            x, y, width, height = result['box']
            height = height / 2
            eye1 = Rectangle((x, y), width / 2, height, fill=False, color='red')
            eye2 = Rectangle(((x + (width / 2)), y), width / 2, height, fill=False, color='red')
            return eye1, eye2

    def extract_eye(self, image, img, image_matrix_given=True):
        if not image_matrix_given:
            image = pyplot.imread(img)
        img = self.return_boxes(self.detector.detect_faces(image))
        if img is None:
            return False
        left = img[0]
        x = int(left.xy[0])
        y = int(left.xy[1])
        xw = int(left.xy[0] + left._width)
        yh = int(left.xy[1] + left._height)
        l_eye = image[y:yh, x:xw]
        l_eye = cv2.resize(l_eye, (256, 256))
        l_eye = self.sharpen(l_eye)
        l_eye = tf.image.rgb_to_grayscale(l_eye)

        right = img[1]
        x = int(right.xy[0])
        y = int(right.xy[1])
        xw = int(right.xy[0] + right._width)
        yh = int(right.xy[1] + right._height)
        r_eye = image[y:yh, x:xw]
        r_eye = cv2.resize(r_eye, (256, 256))
        r_eye = self.sharpen(r_eye)
        r_eye = tf.image.rgb_to_grayscale(r_eye)

        return (l_eye, r_eye), (left, right)

    @staticmethod
    def produce_eye_output(model, inputs):
        outputs = model(inputs)
        print(outputs)
        if tf.argmax(outputs[0]) == 0 or tf.argmax(outputs[1]) == 0:
            return True
        else:
            return False

    def eye_classification(self, l_eye, r_eye, model):
        l_eye = tf.expand_dims(l_eye, axis=0)
        r_eye = tf.expand_dims(r_eye, axis=0)
        print(l_eye.shape, r_eye.shape)
        inputs = tf.concat([l_eye, r_eye], axis=0)
        print(inputs.shape)
        output = self.produce_eye_output(model, inputs)
        if output:
            return True

    @staticmethod
    def yawn_detection(face, model):
        inputs = cv2.resize(face, (256, 256))
        inputs = tf.expand_dims(inputs, axis=0)
        output = model(inputs)
        print(output)
        output = np.argmax(np.round(np.squeeze(output)))
        return output

    def drowsiness(self, image):
        yawn = self.yawn_detection(image, self.models[1])
        eyes = self.extract_eye(image, "")
        markings = False
        if not eyes:
            eye = False
        else:
            eye_class = eyes[0]
            markings = eyes[1]
            eye = self.eye_classification(eye_class[0], eye_class[1], self.models[0])
        if yawn == 0:
            print("Danger Sign : Yawn") # add any alarm sounds
        if eye:
            print("Danger Sign : Eyes Closed")  # add any alarm sounds
        if markings is not False:
            return markings
