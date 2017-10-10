"""
Simple script for analysing Carla rosbag
"""

import rosbag
import numpy as np
import matplotlib.pyplot as plt


def get_base_waypoints(bag):

    messages = list(bag.read_messages(topics="/base_waypoints"))

    waypoints = messages[0].message.waypoints

    x_position = []
    y_position = []

    for waypoint in waypoints:

        position = waypoint.pose.pose.position
        x_position.append(position.x)
        y_position.append(position.y)

    xy_positions = [(x, y) for x, y in zip(x_position, y_position)]

    return np.array(xy_positions)


def get_car_positions(bag):

    messages = list(bag.read_messages(topics="/current_pose"))

    x_position = []
    y_position = []

    for message in messages:
        position = message.message.pose.position
        x_position.append(position.x)
        y_position.append(position.y)

    xy_positions = [(x, y) for x, y in zip(x_position, y_position)]

    return np.array(xy_positions)


def get_final_waypoints(bag):

    messages = list(bag.read_messages(topics="/final_waypoints"))

    final_waypoints = []

    for message in messages:

        waypoints = message.message.waypoints

        x_position = []
        y_position = []

        for waypoint in waypoints:

            position = waypoint.pose.pose.position
            x_position.append(position.x)
            y_position.append(position.y)

        xy_positions = [(x, y) for x, y in zip(x_position, y_position)]
        final_waypoints.append(np.array(xy_positions))

    return final_waypoints


def get_final_waypoints_velocities(bag):

    messages = list(bag.read_messages(topics="/final_waypoints"))

    velocities = []

    for message in messages:

        waypoints = message.message.waypoints

        for waypoint in waypoints:

            velocities.append(waypoint.twist.twist.linear.x)

    return velocities


def main():

    bag_path = "/home/student/data_partition/data/carla/data.bag"

    with rosbag.Bag(bag_path) as bag:

        waypoints = get_base_waypoints(bag)
        car_positions = get_car_positions(bag)
        final_waypoints = get_final_waypoints(bag)
        final_velocities = get_final_waypoints_velocities(bag)

    fig, ax = plt.subplots()

    for waypoints in final_waypoints:

        ax.scatter(waypoints[:, 0], waypoints[:, 1], c='r')

    # for index in range(len(waypoints)):
    #     ax.annotate(str(index), (waypoints[index][0], waypoints[index][1]))

    ax.scatter(car_positions[:, 0], car_positions[:, 1], c='g')
    ax.scatter(waypoints[:, 0], waypoints[:, 1], c='b')

    # for index in range(len(car_positions)):
    #     ax.annotate(str(index), (car_positions[index][0], car_positions[index][1]))

    plt.show()

    print("Final waypoints velocities min, max and std:")
    print(np.min(final_velocities))
    print(np.max(final_velocities))
    print(np.std(final_velocities))

if __name__ == "__main__":

    main()