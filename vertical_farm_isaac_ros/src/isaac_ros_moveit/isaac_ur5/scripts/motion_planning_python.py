#!/usr/bin/env python3
"""
A script to outline the fundamentals of the moveit_py motion planning API.
"""

import time

# generic ros libraries
import rclpy
from rclpy.logging import get_logger

from rclpy.node import Node
# moveit python library
from moveit.core.robot_state import RobotState
from moveit.planning import (
    MoveItPy,
    MultiPipelinePlanRequestParameters,
)
from moveit.core.robot_trajectory import RobotTrajectory
from moveit_msgs.msg import RobotTrajectory as rt
class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(rt, 'traj', 10)
        timer_period = 0.5  # seconds


def plan_and_execute(
    robot,
    planning_component,
    logger,
    minimal_publisher ,
    single_plan_parameters=None,
    multi_plan_parameters=None,
    sleep_time=0.0,
):
    """Helper function to plan and execute a motion."""
    # plan to goal
    logger.info("Planning trajectory")
    if multi_plan_parameters is not None:
        plan_result = planning_component.plan(
            multi_plan_parameters=multi_plan_parameters
        )
    elif single_plan_parameters is not None:
        plan_result = planning_component.plan(
            single_plan_parameters=single_plan_parameters
        )
    else:
        plan_result = planning_component.plan()

    # execute the plan
    if plan_result:
        logger.info("Executing plan")
        robot_trajectory = plan_result.trajectory
        x = robot_trajectory.get_robot_trajectory_msg()
        logger.info("progressive")
        logger.info('Publishing: "%s"' % x.joint_trajectory)
        minimal_publisher.publisher_.publish(x)

        robot.execute(robot_trajectory, controllers=[])
    else:
        logger.error("Planning failed")

    time.sleep(sleep_time)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

def main():

    ###################################################################
    # MoveItPy Setup
    ###################################################################
    rclpy.init()
    minimal_publisher = MinimalPublisher()
    logger = get_logger("moveit_py.pose_goal")

    # instantiate MoveItPy instance and get planning component
    panda = MoveItPy(node_name="moveit_py")
    panda_arm = panda.get_planning_component("manipulator")
    logger.info("MoveItPy instance created")

    ###########################################################################
    # Plan 2 - set goal state with RobotState object
    ###########################################################################
    # set pose goal with PoseStamped message
    from geometry_msgs.msg import PoseStamped
    panda_arm.set_start_state_to_current_state()
    pose_goal = PoseStamped()
    pose_goal.header.frame_id = "base_link"
    pose_goal.pose.orientation.x = 1e-6
    pose_goal.pose.orientation.y = 1e-6
    pose_goal.pose.orientation.z = 1e-6
    pose_goal.pose.orientation.w = 1.000000
    pose_goal.pose.position.x = 1e-6
    pose_goal.pose.position.y = 0.24444
    pose_goal.pose.position.z = 0.46
    panda_arm.set_goal_state(pose_stamped_msg=pose_goal, pose_link="wrist_3_link")

    # plan to goal
    plan_and_execute(panda, panda_arm, logger,minimal_publisher , sleep_time=3.0)



if __name__ == "__main__":
    main()
