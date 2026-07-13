#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
capture_one_view_image.py
Script dedicated to capturing front-view RGB images
Adapted from capture_three_view_images.py
"""

import rospy
import cv2
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from collections import deque


class OneViewImageCapture:
    def __init__(self):
        self.bridge = CvBridge()

        # Image queues
        self.img_front_deque = deque(maxlen=10)

        # Image topics
        self.front_topic = '/camera_f/color/image_raw'

        # Initialize ROS node
        rospy.init_node('one_view_capture', anonymous=True)

        # Subscribe to image topics
        rospy.Subscriber(self.front_topic, Image, self.img_front_callback, queue_size=10)

    def img_front_callback(self, msg):
        """Front-view camera callback."""
        if len(self.img_front_deque) >= 10:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def get_latest_frame(self):
        """Get the latest front-view image."""
        if len(self.img_front_deque) == 0:
            return None

        front_msg = self.img_front_deque[-1]

        # Convert to OpenCV format
        try:
            front_img = self.bridge.imgmsg_to_cv2(front_msg, 'bgr8')
        except Exception as e:
            print(f"Image conversion failed: {e}")
            return None

        return front_img

    def save_image(self, front_img, output_dir='.'):
        """Save front-view image."""
        os.makedirs(output_dir, exist_ok=True)

        front_path = os.path.join(output_dir, 'front_view.jpg')
        cv2.imwrite(front_path, front_img)

        print("Images saved:")
        print(f"  Front view: {front_path}")

        return front_path

    def capture_once(self, output_dir='.'):
        """Capture front-view image once."""
        print("Starting front-view image capture...")

        try:
            msg = rospy.wait_for_message(self.front_topic, Image, timeout=2.0)
        except rospy.ROSException:
            print("Unable to get front view image data")
            return None

        try:
            front_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            print(f"Image conversion failed: {e}")
            return None

        return self.save_image(front_img, output_dir)

    def run(self, output_dir='.'):
        """Main run loop."""
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
    """Main entry point."""
    output_dir = _default_tmp_dir()

    capture = OneViewImageCapture()

    success = capture.run(output_dir)

    if not success:
        print("\nImage capture failed!")
        print("Please check:")
        print("  1. ROS environment is set up (source /opt/ros/noetic/setup.bash)")
        print("  2. Camera node is started (roslaunch realsense2_camera multi_camera.launch)")
        print("  3. Camera topics are publishing correctly")

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
