import time
import argparse
import subprocess
import os
from os.path import join

import numpy as np
import rospy
import rospkg

from gazebo_simulation import GazeboSimulation

INIT_POSITION = [-2, 3, 1.57]  # in world frame
GOAL_POSITION = [20.5, 28]  # relative to the initial position

def compute_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def path_coord_to_gazebo_coord(x, y):
    RADIUS = 0.075
    r_shift = -RADIUS - (30 * RADIUS * 2)
    c_shift = RADIUS + 5

    gazebo_x = x * (RADIUS * 2) + r_shift
    gazebo_y = y * (RADIUS * 2) + c_shift

    return (gazebo_x, gazebo_y)

def run_navigation_test():
    try:
        rospy.init_node('gym', anonymous=True)  # , log_level=rospy.FATAL)
        rospy.set_param('/use_sim_time', True)

        gazebo_sim = GazeboSimulation(init_position=INIT_POSITION)

        init_coor = (INIT_POSITION[0], INIT_POSITION[1])
        goal_coor = (INIT_POSITION[0] + GOAL_POSITION[0], INIT_POSITION[1] + GOAL_POSITION[1])

        pos = gazebo_sim.get_model_state().pose.position
        curr_coor = (pos.x, pos.y)
        collided = True

        while compute_distance(init_coor, curr_coor) > 0.1 or collided:
            gazebo_sim.reset()  # Reset to the initial position
            pos = gazebo_sim.get_model_state().pose.position
            curr_coor = (pos.x, pos.y)
            collided = gazebo_sim.get_hard_collision()
            time.sleep(1)

        launch_file = join(base_path, '..', 'jackal_helper/launch/move_base_TEB.launch')
        nav_stack_process = subprocess.Popen([
            'roslaunch',
            launch_file,
        ])

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

        curr_time = rospy.get_time()
        pos = gazebo_sim.get_model_state().pose.position
        curr_coor = (pos.x, pos.y)

        while compute_distance(init_coor, curr_coor) < 0.1:
            curr_time = rospy.get_time()
            pos = gazebo_sim.get_model_state().pose.position
            curr_coor = (pos.x, pos.y)
            time.sleep(0.01)

        start_time = curr_time
        start_time_cpu = time.time()
        collided = False

        while compute_distance(goal_coor, curr_coor) > 1 and not collided and curr_time - start_time < 100:
            curr_time = rospy.get_time()
            pos = gazebo_sim.get_model_state().pose.position
            curr_coor = (pos.x, pos.y)
            print("Time: %.2f (s), x: %.2f (m), y: %.2f (m)" % (curr_time - start_time, *curr_coor), end="\r")
            collided = gazebo_sim.get_hard_collision()
            while rospy.get_time() - curr_time < 0.1:
                time.sleep(0.01)

        success = False
        if collided:
            status = "collided"
            actual_time = 200.0  # Penalized time for collision
        elif curr_time - start_time >= 100:
            status = "timeout"
            actual_time = 200.0  # Penalized time for timeout
        else:
            status = "succeeded"
            success = True
            actual_time = curr_time - start_time

        print("Navigation %s with time %.4f (s)" % (status, actual_time))

        if args.world_idx >= 300:  # DynaBARN environment which does not have a planned path
            path_length = GOAL_POSITION[0] - INIT_POSITION[0]
        else:
            path_file_name = join(base_path, "worlds/BARN/path_files", "path_%d.npy" % args.world_idx)
            path_array = np.load(path_file_name)
            path_array = [path_coord_to_gazebo_coord(*p) for p in path_array]
            path_array = np.insert(path_array, 0, (INIT_POSITION[0], INIT_POSITION[1]), axis=0)
            path_array = np.insert(path_array, len(path_array), (INIT_POSITION[0] + GOAL_POSITION[0], INIT_POSITION[1] + GOAL_POSITION[1]), axis=0)
            path_length = 0
            for p1, p2 in zip(path_array[:-1], path_array[1:]):
                path_length += compute_distance(p1, p2)

        optimal_time = path_length / 2
        nav_metric = int(success) * optimal_time / np.clip(actual_time, 2 * optimal_time, 8 * optimal_time)
        print("Navigation metric: %.4f" % (nav_metric))

        nav_stack_process.terminate()
        nav_stack_process.wait()

        return actual_time, nav_metric

    except Exception as e:
        print(f"Error during navigation test: {e}")
        return 200.0, 0.0  # Penalized time and zero metric in case of error

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test BARN navigation challenge')
    parser.add_argument('--world_idx', type=int, default=0)
    parser.add_argument('--gui', action="store_true")
    parser.add_argument('--out', type=str, default="out.txt")
    args = parser.parse_args()

    os.environ["JACKAL_LASER"] = "1"
    os.environ["JACKAL_LASER_MODEL"] = "ust10"
    os.environ["JACKAL_LASER_OFFSET"] = "-0.065 0 0.01"

    if args.world_idx < 300:  # static environment from 0-299
        world_name = "BARN/world_%d.world" % (args.world_idx)
        INIT_POSITION = [-2.25, 3, 1.57]  # in world frame
        GOAL_POSITION = [10, 4]  # relative to the initial position
    elif args.world_idx < 360:  # Dynamic environment from 300-359
        world_name = "DynaBARN/world_%d.world" % (args.world_idx - 300)
        INIT_POSITION = [11, 0, 3.14]  # in world frame
        GOAL_POSITION = [-20, 0]  # relative to the initial position
    else:
        raise ValueError("World index %d does not exist" % args.world_idx)

    print(">>>>>>>>>>>>>>>>>> Loading Gazebo Simulation with %s <<<<<<<<<<<<<<<<<<" % (world_name))
    rospack = rospkg.RosPack()
    base_path = rospack.get_path('jackal_helper')
    os.environ['GAZEBO_PLUGIN_PATH'] = os.path.join(base_path, "plugins")

    total_time = 0.0
    total_metric = 0.0

    for i in range(1):
        print(f"Starting navigation test {i+1}/10")
        
        # Relaunch Gazebo Simulation for each iteration
        launch_file = join(base_path, 'launch', 'gazebo_launch_laser.launch')
        world_name = join(base_path, "worlds", world_name)

        gazebo_process = subprocess.Popen([
            'roslaunch',
            launch_file,
            'world_name:=' + world_name,
            'gui:=' + ("true" if args.gui else "false")
        ])
        time.sleep(5)  # sleep to wait until the gazebo being created

        actual_time, nav_metric = run_navigation_test()
        total_time += actual_time
        total_metric += nav_metric

        gazebo_process.terminate()
        gazebo_process.wait()

    avg_time = total_time / 10.0
    avg_metric = total_metric / 10.0

    print("Average navigation time: %.4f (s)" % avg_time)
    print("Average navigation metric: %.4f" % avg_metric)

    with open(args.out, "a") as f:
        f.write("Average time: %.4f (s), Average metric: %.4f\n" % (avg_time, avg_metric))

