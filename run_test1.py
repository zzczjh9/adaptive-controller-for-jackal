import time
import argparse
import subprocess
import os
from os.path import join

import numpy as np
import rospy
import rospkg
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from gazebo_simulation import GazeboSimulation

INIT_POSITION = [-2, 3, 1.57]  # in world frame
GOAL_POSITION = [0, 10]  # relative to the initial position

teb_velocities = []
teb_last_timestep = None
last_scan_data = None


def compute_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def path_coord_to_gazebo_coord(x, y):
    RADIUS = 0.075
    r_shift = -RADIUS - (30 * RADIUS * 2)
    c_shift = RADIUS + 5

    gazebo_x = x * (RADIUS * 2) + r_shift
    gazebo_y = y * (RADIUS * 2) + c_shift

    return (gazebo_x, gazebo_y)


def cmd_vel_callback(msg):
    global teb_velocities, teb_last_timestep
    teb_time = rospy.get_time()
    teb_velocities.append((teb_time, msg.linear.x, msg.linear.y, msg.angular.z))
    teb_last_timestep = (teb_time, msg.linear.x, msg.linear.y, msg.angular.z)


def scan_callback(msg):
    global last_scan_data
    last_scan_data = msg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test BARN navigation challenge')
    parser.add_argument('--world_idx', type=int, default=0, help='Index of the world to load')
    parser.add_argument('--gui', action="store_true", help='Launch Gazebo with GUI')
    parser.add_argument('--laser', action="store_true", help='Enable the front laser sensor')
    parser.add_argument('--out', type=str, default="out.txt", help='Output file for navigation results')
    args = parser.parse_args()

    ##########################################################################################
    ## 0. Launch Gazebo Simulation
    ##########################################################################################

    # Set environment variables based on the laser argument
    if args.laser:
        os.environ["JACKAL_LASER"] = "1"
        os.environ["JACKAL_LASER_MODEL"] = "ust10"
        os.environ["JACKAL_LASER_OFFSET"] = "-0.065 0 0.01"
    else:
        os.environ["JACKAL_LASER"] = "0"

    if args.world_idx < 300:  # static environment from 0-299
        world_name = "BARN/world_%d.world" % args.world_idx
        INIT_POSITION = [-2.25, 3, 1.57]
        GOAL_POSITION = [0, 10]
    elif args.world_idx < 360:  # Dynamic environment from 300-359
        world_name = "DynaBARN/world_%d.world" % (args.world_idx - 300)
        INIT_POSITION = [11, 0, 3.14]
        GOAL_POSITION = [-20, 0]
    else:
        raise ValueError("World index %d does not exist" % args.world_idx)

    print(">>>>>>>>>>>>>>>>>> Loading Gazebo Simulation with %s <<<<<<<<<<<<<<<<<<" % world_name)
    rospack = rospkg.RosPack()
    base_path = rospack.get_path('jackal_helper')
    os.environ['GAZEBO_PLUGIN_PATH'] = os.path.join(base_path, "plugins")

    launch_file = join(base_path, 'launch', 'gazebo_launch.launch')
    world_name = join(base_path, "worlds", world_name)

    # Pass the laser config as an argument to the launch file
    gazebo_process = subprocess.Popen([
        'roslaunch',
        launch_file,
        'world_name:=' + world_name,
        'config:=' + ('front_laser' if args.laser else 'base'),
        'gui:=' + ("true" if args.gui else "false")
    ])

    time.sleep(5)  # sleep to wait until the gazebo being created

    rospy.init_node('gym', anonymous=True)
    rospy.set_param('/use_sim_time', True)

    # GazeboSimulation provides useful interface to communicate with gazebo
    gazebo_sim = GazeboSimulation(init_position=INIT_POSITION)

    init_coor = (INIT_POSITION[0], INIT_POSITION[1])
    goal_coor = (INIT_POSITION[0] + GOAL_POSITION[0], INIT_POSITION[1] + GOAL_POSITION[1])

    pos = gazebo_sim.get_model_state().pose.position
    curr_coor = (pos.x, pos.y)
    collided = True

    # Subscribe to /cmd_vel to get velocities from TEB planner
    rospy.Subscriber('/cmd_vel', Twist, cmd_vel_callback)

    # Subscribe to /scan to get laser scan data
    rospy.Subscriber('/front/scan', LaserScan, scan_callback)

    # check whether the robot is reset, the collision is False
    while compute_distance(init_coor, curr_coor) > 0.1 or collided:
        gazebo_sim.reset()  # Reset to the initial position
        pos = gazebo_sim.get_model_state().pose.position
        curr_coor = (pos.x, pos.y)
        collided = gazebo_sim.get_hard_collision()
        time.sleep(1)

    ##########################################################################################
    ## 1. Launch your navigation stack
    ## (Customize this block to add your own navigation stack)
    ##########################################################################################

    launch_file = join(base_path, '..', 'jackal_helper/launch/move_base_DWA.launch')
    nav_stack_process = subprocess.Popen([
        'roslaunch',
        launch_file,
    ])

    # Make sure your navigation stack receives the correct goal position defined in GOAL_POSITION
    import actionlib
    from geometry_msgs.msg import Quaternion
    from move_base_msgs.msg import MoveBaseGoal, MoveBaseAction

    nav_as = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
    mb_goal = MoveBaseGoal()
    mb_goal.target_pose.header.frame_id = 'odom'
    mb_goal.target_pose.pose.position.x = GOAL_POSITION[0]
    mb_goal.target_pose.pose.position.y = GOAL_POSITION[1]
    mb_goal.target_pose.pose.position.z = 0
    mb_goal.target_pose.pose.orientation = Quaternion(0, 0, 0, 1)

    nav_as.wait_for_server()
    nav_as.send_goal(mb_goal)

    ##########################################################################################
    ## 2. Start navigation
    ##########################################################################################

    curr_time = rospy.get_time()
    pos = gazebo_sim.get_model_state().pose.position
    curr_coor = (pos.x, pos.y)

    # check whether the robot started to move
    while compute_distance(init_coor, curr_coor) < 0.1:
        curr_time = rospy.get_time()
        pos = gazebo_sim.get_model_state().pose.position
        curr_coor = (pos.x, pos.y)
        time.sleep(0.01)

    start_time = curr_time
    start_time_cpu = time.time()
    collided = False

    # Initialize variables to store last position for velocity calculation
    last_time = rospy.get_time()
    last_pos = gazebo_sim.get_model_state().pose.position

    while compute_distance(goal_coor, curr_coor) > 1 and not collided and curr_time - start_time < 100:
        curr_time = rospy.get_time()
        pos = gazebo_sim.get_model_state().pose.position
        curr_coor = (pos.x, pos.y)
        print("Time: %.2f (s), x: %.2f (m), y: %.2f (m)" % (curr_time - start_time, *curr_coor), end="\r")
        collided = gazebo_sim.get_hard_collision()

        # Calculate actual velocities
        actual_linear_velocity = compute_distance((last_pos.x, last_pos.y), (pos.x, pos.y)) / (curr_time - last_time)
        actual_angular_velocity = abs(gazebo_sim.get_model_state().pose.orientation.z - last_pos.z) / (
                    curr_time - last_time)

        # Print TEB requested velocities, actual velocities, and scan data for the last step
        #if teb_last_timestep:
        #    teb_time, teb_linear_x, teb_linear_y, teb_angular_z = teb_last_timestep
        #    print(
        #        f"TEB Requested Velocities at {teb_time} s: linear_x={teb_linear_x}, linear_y={teb_linear_y}, angular_z={teb_angular_z}")
        #    print(
        #        f"Actual Velocities at {curr_time} s: linear={actual_linear_velocity}, angular={actual_angular_velocity}")
        #    if last_scan_data:
        #        print(f"Scan data at {teb_time} s: ranges={last_scan_data.ranges}")

        last_time = curr_time
        last_pos = pos

        while rospy.get_time() - curr_time < 0.1:
            time.sleep(0.01)

    ##########################################################################################
    ## 3. Report metrics and generate log
    ##########################################################################################

    print(">>>>>>>>>>>>>>>>>> Test finished! <<<<<<<<<<<<<<<<<<")
    success = False
    if collided:
        status = "collided"
    elif curr_time - start_time >= 100:
        status = "timeout"
    else:
        status = "succeeded"
        success = True
    print("Navigation %s with time %.4f (s)" % (status, curr_time - start_time))

    if args.world_idx >= 300:  # DynaBARN environment which does not have a planned path
        path_length = GOAL_POSITION[0] - INIT_POSITION[0]
    else:
        path_file_name = join(base_path, "worlds/BARN/path_files", "path_%d.npy" % args.world_idx)
        path_array = np.load(path_file_name)
        path_array = [path_coord_to_gazebo_coord(*p) for p in path_array]
        path_array = np.insert(path_array, 0, (INIT_POSITION[0], INIT_POSITION[1]), axis=0)
        path_array = np.insert(path_array, len(path_array),
                               (INIT_POSITION[0] + GOAL_POSITION[0], INIT_POSITION[1] + GOAL_POSITION[1]), axis=0)
        path_length = 0
        for p1, p2 in zip(path_array[:-1], path_array[1:]):
            path_length += compute_distance(p1, p2)

    # Navigation metric: 1_success *  optimal_time / clip(actual_time, 2 * optimal_time, 8 * optimal_time)
    optimal_time = path_length / 2
    actual_time = curr_time - start_time
    nav_metric = int(success) * optimal_time / np.clip(actual_time, 2 * optimal_time, 8 * optimal_time)
    print("Navigation metric: %.4f" % nav_metric)

    with open(args.out, "a") as f:
        f.write("%d %d %d %d %.4f %.4f\n" % (
        args.world_idx, success, collided, (curr_time - start_time) >= 100, curr_time - start_time, nav_metric))

    gazebo_process.terminate()
    gazebo_process.wait()
    nav_stack_process.terminate()
    nav_stack_process.wait()
