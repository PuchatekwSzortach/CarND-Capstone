"""
Tests for waypoints helper
"""

import styx_msgs.msg
import geometry_msgs.msg
import numpy as np

import src.waypoint_updater.waypoints_helper as waypoints_helper


def test_get_distance_between_points():

    first = geometry_msgs.msg.Point()
    first.x = 0
    first.y = 0

    second = geometry_msgs.msg.Point()
    second.x = 10
    second.y = 20

    expected = np.sqrt(500)
    actual = waypoints_helper.get_distance_between_points(first, second)

    assert np.isclose(expected, actual)


def test_get_closest_waypoint_index():

    pose = geometry_msgs.msg.Pose()
    pose.position.x = 10
    pose.position.y = 10

    waypoints_matrix = np.array([
        [0, 0],
        [8, 8],
        [20, 20]
    ])

    assert 1 == waypoints_helper.get_closest_waypoint_index(pose.position, waypoints_matrix)


def test_get_sublist_simple():

    elements = [1, 2, 3, 4, 5]
    start_index = 1
    size = 2

    expected = [2, 3]
    actual = waypoints_helper.get_sublist(elements, start_index, size)

    assert expected == actual


def test_get_sublist_wrapped():

    elements = [1, 2, 3, 4, 5]
    start_index = 3
    size = 3

    expected = [4, 5, 1]
    actual = waypoints_helper.get_sublist(elements, start_index, size)

    assert expected == actual