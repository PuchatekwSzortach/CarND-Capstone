from styx_msgs.msg import TrafficLight
import numpy as np
import cv2
from keras.models import model_from_json
import tensorflow as tf
import json
import rospy

class TLClassifier(object):

    def __init__(self):
        # Lower and Upper threshold for color extraction
        self.lower = np.array([150, 100, 150])
        self.upper = np.array([180, 255, 255])

        with open('light_classification/saved_models/model128.json', 'r') as f:
            loaded_model_json = f.read()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights('light_classification/saved_models/weights128.hdf5')
        self.graph = tf.get_default_graph()

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # state = TrafficLight.UNKNOWN

        # red_area, red_image = self.get_colored_area(image, self.lower, self.upper)

        # Needs careful tuning for the number of red pixesls
        # red_pixels = 40

        # if red_area > red_pixels:
        #     state = TrafficLight.RED

        # return state


        image = self.process_image(image)

        with self.graph.as_default():
            pred_class = self.model.predict_classes(image, batch_size = 1, verbose = 0)
            # rospy.logwarn("Predicted class:{}".format(pred_class))
            if pred_class == 0:
                state = TrafficLight.RED
            elif pred_class == 1:
                state = TrafficLight.YELLOW
            elif pred_class == 2:
                state = TrafficLight.GREEN
            else:
                state = TrafficLight.UNKNOWN
        return state

        
    def get_colored_area(self, image, lower, upper):

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask_image = cv2.inRange(hsv_image, lower, upper)
        extracted_image = cv2.bitwise_and(image, image, mask=mask_image)
        area = cv2.countNonZero(mask_image)

        return area, extracted_image


    def process_image(self, image):

        desired_shape = (128, 128)
        image = cv2.resize(image, desired_shape, cv2.INTER_LINEAR)
        image = image.astype('float32') / 255
        processed_image = image.reshape(1, *image.shape)
        return processed_image 

