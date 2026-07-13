#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
capture_three_view_rgb_depth.py
Script for capturing 3-view RGB+Depth images
Adapted from a local HDF5 collection script; retarget topics for your cameras.
"""

import os
from collections import deque

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class ThreeViewRgbDepthCapture:
    def __init__(self):
        self.bridge = CvBridge()

        # RGB image queues
        self.rgb_front_deque = deque(maxlen=10)
        self.rgb_left_deque = deque(maxlen=10)
        self.rgb_right_deque = deque(maxlen=10)

        # Depth image queues
        self.depth_front_deque = deque(maxlen=10)
        self.depth_left_deque = deque(maxlen=10)
        self.depth_right_deque = deque(maxlen=10)

        # Image topics
        self.rgb_front_topic = '/camera_f/color/image_raw'
        self.rgb_left_topic = '/camera_l/color/image_raw'
        self.rgb_right_topic = '/camera_r/color/image_raw'

        self.depth_front_topic = '/camera_f/depth/image_raw'
        self.depth_left_topic = '/camera_l/depth/image_rect_raw'
        self.depth_right_topic = '/camera_r/depth/image_rect_raw'

        # Initialize ROS node
        rospy.init_node('three_view_rgb_depth_capture', anonymous=True)

        # Subscribe to RGB topics
        rospy.Subscriber(self.rgb_front_topic, Image, self.rgb_front_callback, queue_size=10)
        rospy.Subscriber(self.rgb_left_topic, Image, self.rgb_left_callback, queue_size=10)
        rospy.Subscriber(self.rgb_right_topic, Image, self.rgb_right_callback, queue_size=10)

        # Subscribe to depth topics
        rospy.Subscriber(self.depth_front_topic, Image, self.depth_front_callback, queue_size=10)
        rospy.Subscriber(self.depth_left_topic, Image, self.depth_left_callback, queue_size=10)
        rospy.Subscriber(self.depth_right_topic, Image, self.depth_right_callback, queue_size=10)

    def rgb_front_callback(self, msg):
        if len(self.rgb_front_deque) >= 10:
            self.rgb_front_deque.popleft()
        self.rgb_front_deque.append(msg)

    def rgb_left_callback(self, msg):
        if len(self.rgb_left_deque) >= 10:
            self.rgb_left_deque.popleft()
        self.rgb_left_deque.append(msg)

    def rgb_right_callback(self, msg):
        if len(self.rgb_right_deque) >= 10:
            self.rgb_right_deque.popleft()
        self.rgb_right_deque.append(msg)

    def depth_front_callback(self, msg):
        if len(self.depth_front_deque) >= 10:
            self.depth_front_deque.popleft()
        self.depth_front_deque.append(msg)

    def depth_left_callback(self, msg):
        if len(self.depth_left_deque) >= 10:
            self.depth_left_deque.popleft()
        self.depth_left_deque.append(msg)

    def depth_right_callback(self, msg):
        if len(self.depth_right_deque) >= 10:
            self.depth_right_deque.popleft()
        self.depth_right_deque.append(msg)

    def get_synchronized_frames(self):
        if (
            len(self.rgb_front_deque) == 0
            or len(self.rgb_left_deque) == 0
            or len(self.rgb_right_deque) == 0
            or len(self.depth_front_deque) == 0
            or len(self.depth_left_deque) == 0
            or len(self.depth_right_deque) == 0
        ):
            return None

        rgb_front_time = self.rgb_front_deque[-1].header.stamp.to_sec()
        rgb_left_time = self.rgb_left_deque[-1].header.stamp.to_sec()
        rgb_right_time = self.rgb_right_deque[-1].header.stamp.to_sec()
        depth_front_time = self.depth_front_deque[-1].header.stamp.to_sec()
        depth_left_time = self.depth_left_deque[-1].header.stamp.to_sec()
        depth_right_time = self.depth_right_deque[-1].header.stamp.to_sec()

        times = [
            rgb_front_time,
            rgb_left_time,
            rgb_right_time,
            depth_front_time,
            depth_left_time,
            depth_right_time,
        ]
        max_time_diff = max(times) - min(times)

        if max_time_diff > 0.1:
            print(f"Images not synchronized: time diff {max_time_diff:.3f} s")
            return None

        rgb_front_msg = self.rgb_front_deque[-1]
        rgb_left_msg = self.rgb_left_deque[-1]
        rgb_right_msg = self.rgb_right_deque[-1]
        depth_front_msg = self.depth_front_deque[-1]
        depth_left_msg = self.depth_left_deque[-1]
        depth_right_msg = self.depth_right_deque[-1]

        try:
            rgb_front = self.bridge.imgmsg_to_cv2(rgb_front_msg, 'bgr8')
            rgb_left = self.bridge.imgmsg_to_cv2(rgb_left_msg, 'bgr8')
            rgb_right = self.bridge.imgmsg_to_cv2(rgb_right_msg, 'bgr8')

            depth_front = self.bridge.imgmsg_to_cv2(depth_front_msg, 'passthrough')
            depth_left = self.bridge.imgmsg_to_cv2(depth_left_msg, 'passthrough')
            depth_right = self.bridge.imgmsg_to_cv2(depth_right_msg, 'passthrough')
        except Exception as e:
            print(f"Image conversion failed: {e}")
            return None

        depth_front = self._normalize_depth(depth_front)
        depth_left = self._normalize_depth(depth_left)
        depth_right = self._normalize_depth(depth_right)

        return rgb_front, rgb_left, rgb_right, depth_front, depth_left, depth_right

    def _normalize_depth(self, depth_img):
        # Convert float depth (meters) to uint16 (millimeters) for PNG save
        if depth_img.dtype == 'float32' or depth_img.dtype == 'float64':
            depth_mm = depth_img * 1000.0
            depth_mm[depth_mm < 0] = 0
            depth_mm[depth_mm > 65535] = 65535
            return depth_mm.astype('uint16')
        return depth_img

    def save_images(self, rgb_front, rgb_left, rgb_right, depth_front, depth_left, depth_right, output_dir='.'):
        os.makedirs(output_dir, exist_ok=True)

        front_rgb_path = os.path.join(output_dir, 'front_view.png')
        left_rgb_path = os.path.join(output_dir, 'left_view.png')
        right_rgb_path = os.path.join(output_dir, 'right_view.png')

        front_depth_path = os.path.join(output_dir, 'front_depth.png')
        left_depth_path = os.path.join(output_dir, 'left_depth.png')
        right_depth_path = os.path.join(output_dir, 'right_depth.png')

        cv2.imwrite(front_rgb_path, rgb_front)
        cv2.imwrite(left_rgb_path, rgb_left)
        cv2.imwrite(right_rgb_path, rgb_right)

        cv2.imwrite(front_depth_path, depth_front)
        cv2.imwrite(left_depth_path, depth_left)
        cv2.imwrite(right_depth_path, depth_right)

        print("Images saved:")
        print(f"  Front RGB: {front_rgb_path}")
        print(f"  Left RGB: {left_rgb_path}")
        print(f"  Right RGB: {right_rgb_path}")
        print(f"  Front Depth: {front_depth_path}")
        print(f"  Left Depth: {left_depth_path}")
        print(f"  Right Depth: {right_depth_path}")

        return (
            front_rgb_path,
            left_rgb_path,
            right_rgb_path,
            front_depth_path,
            left_depth_path,
            right_depth_path,
        )

    def capture_once(self, output_dir='.'):
        print("Starting three-view RGB+Depth capture...")

        for _ in range(50):
            frames = self.get_synchronized_frames()
            if frames is not None:
                return self.save_images(*frames, output_dir=output_dir)

            rospy.sleep(0.1)

        print("Unable to get synchronized image data")
        return None

    def run(self, output_dir='.'):
        try:
            result = self.capture_once(output_dir)

            if result:
                return True
            print("\nCapture failed!")
            return False

        except rospy.ROSInterruptException:
            print("ROS interrupted")
            return False
        except Exception as e:
            print(f"Error during capture: {e}")
            return False


def _default_tmp_dir():
    env = os.environ.get("THREE_VIEW_TMP_DIR", "").strip()
    if env:
        return env
    skill_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(skill_dir, "tmp")


def main():
    output_dir = _default_tmp_dir()

    capture = ThreeViewRgbDepthCapture()
    success = capture.run(output_dir)

    if not success:
        print("\nRGB+Depth image capture failed!")
        print("Please check:")
        print("  1. ROS environment is set up (source /opt/ros/noetic/setup.bash)")
        print("  2. Camera node is started (roslaunch realsense2_camera multi_camera.launch)")
        print("  3. Camera topics are publishing correctly")

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
