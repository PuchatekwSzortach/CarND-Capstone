#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifierCV, TLClassifier
import tf
import tf.transformations
import cv2
import tf_helper
import numpy as np
import yaml
import geometry_msgs.msg


STATE_COUNT_THRESHOLD = 3


class TLDetector(object):

    def __init__(self):
        rospy.init_node('tl_detector')

        self.car_pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.traffic_positions = tf_helper.get_given_traffic_lights()

        self.last_traffic_light_state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.last_reported_traffic_light_id = None
        self.last_reported_traffic_light_time = None

        self.traffic_lights = None
        self.image = None

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.bridge = CvBridge()

        self.experiment_environment = rospy.get_param('/experiment_environment', "site")
        self.light_classifier = TLClassifier(self.experiment_environment)
        # self.light_classifier = TLClassifierCV()

        self.listener = tf.TransformListener()

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)

        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb, queue_size=1)
        rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1)

        self.upcoming_stop_light_pub = rospy.Publisher(
            '/upcoming_stop_light_position', geometry_msgs.msg.Point, queue_size=1)

        rospy.spin()

    def pose_cb(self, msg):

        self.car_pose = msg.pose

        # For debugging(Ground Truth data)
        # arguments = [self.traffic_lights, self.car_pose, self.waypoints, self.image]
        arguments = [self.traffic_positions, self.car_pose, self.waypoints, self.image]
        are_arguments_available = all([x is not None for x in arguments])

        if are_arguments_available:

            # Get closest traffic light
            traffic_light = tf_helper.get_closest_traffic_light_ahead_of_car(
                self.traffic_positions.lights, self.car_pose.position, self.waypoints)

            # These values seem so be wrong - Udacity keeps on putting in config different values that what camera
            # actually publishes.
            # image_width = self.config["camera_info"]["image_width"]
            # image_height = self.config["camera_info"]["image_height"]

            # Therefore simply check image size
            self.camera_image = self.image
            self.camera_image.encoding = "rgb8"
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

            traffic_light_state = self.light_classifier.get_classification(cv_image)

            # lights_map = {0: "Red", 1: "Yellow", 2: "Green"}
            # rospy.logwarn("Detected light: {}".format(lights_map.get(traffic_light_state, "Other")))

            if traffic_light_state == TrafficLight.RED or traffic_light == TrafficLight.YELLOW:
                self.upcoming_stop_light_pub.publish(traffic_light.pose.pose.position)

    def waypoints_cb(self, lane):
        self.waypoints = lane.waypoints

    def traffic_cb(self, msg):
        self.traffic_lights = msg.lights

    def image_cb(self, msg):
        self.image = msg


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
