"""
Utilities for traffic light module
"""

import numpy as np
from styx_msgs.msg import TrafficLightArray, TrafficLight
from traffic_light_config import config

def get_given_traffic_lights():
    """
    Return given traffic light positions
    :return: TrafficLightArray
    """
    traffic_lights = TrafficLightArray()
	
    traffic_light_list = []
    traffic_light_positions = config.light_positions
	
    for traffic_light_index, traffic_light_position in enumerate(traffic_light_positions):
	traffic_light = TrafficLight()

	traffic_light.pose.pose.position.x = traffic_light_position[0]
	traffic_light.pose.pose.position.y = traffic_light_position[1]
	traffic_light.pose.pose.position.z = 5.837643
	traffic_light.state = TrafficLight.UNKNOWN
	traffic_light_list.append(traffic_light)
	    
    traffic_lights.lights = traffic_light_list

    return traffic_lights

def get_distance_between_points(first, second):
    """
    Return distance between two points
    :param first: geometry_msgs.msgs.Point instance
    :param second: geometry_msgs.msgs.Point instance
    :return: float
    """

    x_difference = first.x - second.x
    y_difference = first.y - second.y

    return np.sqrt(x_difference**2 + y_difference**2)


def get_waypoints_matrix(waypoints):
    """
    Converts waypoints listt to numpy matrix
    :param waypoints: list of styx_msgs.msg.Waypoint instances
    :return: 2D numpy array
    """

    waypoints_matrix = np.zeros(shape=(len(waypoints), 2), dtype=np.float32)

    for index, waypoint in enumerate(waypoints):
        waypoints_matrix[index, 0] = waypoint.pose.pose.position.x
        waypoints_matrix[index, 1] = waypoint.pose.pose.position.y

    return waypoints_matrix


def get_closest_waypoint_index(position, waypoints_matrix):
    """
    Given a pose and waypoints list, return index of waypoint closest to pose
    :param position: geometry_msgs.msgs.Position instance
    :param waypoints_matrix: numpy matrix with waypoints coordinates
    :return: integer index
    """

    x_distances = waypoints_matrix[:, 0] - position.x
    y_distances = waypoints_matrix[:, 1] - position.y

    squared_distances = x_distances**2 + y_distances**2
    return np.argmin(squared_distances)


def get_info_about_closest_traffic_light_ahead_of_car(traffic_lights, car_position, waypoints_matrix):
    """
    Given list of traffic lights, car position and waypoints, return closest traffic light
    ahead of the car and index of closest waypoint
    :param traffic_lights: list of styx_msgs.msg.TrafficLight instances
    :param car_position: geometry_msgs.msgs.Pose instance
    :param waypoints_matrix: numpy matrix with waypoints coordinates
    :return: styx_msgs.msg.TrafficLight instance
    """
    
    car_waypoint_index = get_closest_waypoint_index(car_position, waypoints_matrix)

    lights_waypoints_indices = []

    for traffic_light_index, traffic_light in enumerate(traffic_lights):

        waypoint_index = get_closest_waypoint_index(traffic_light.pose.pose.position, waypoints_matrix)

        if waypoint_index > car_waypoint_index:
            lights_waypoints_indices.append((traffic_light_index, waypoint_index))

    sorted_traffic_lights_waypoint_indices = sorted(lights_waypoints_indices, key=lambda x: x[1])
    
    if len(lights_waypoints_indices) == 0:
    	light_index = 0
        light_waypoint_index = 0
    else:
        light_index = sorted_traffic_lights_waypoint_indices[0][0]
        light_waypoint_index = sorted_traffic_lights_waypoint_indices[0][1]

    return traffic_lights[light_index], light_waypoint_index