#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
capture_three_view_images.py
Script dedicated to capturing 3-view RGB images
Adapted from a local HDF5 collection script; retarget topics for your cameras.
"""

import rospy
import cv2
import os
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from collections import deque


class ThreeViewImageCapture:
    def __init__(self):
        self.bridge = CvBridge()
        
        # Image queues
        self.img_front_deque = deque(maxlen=10)
        self.img_left_deque = deque(maxlen=10)
        self.img_right_deque = deque(maxlen=10)
        
        # Image topics
        self.front_topic = '/camera_f/color/image_raw'
        self.left_topic = '/camera_l/color/image_raw'
        self.right_topic = '/camera_r/color/image_raw'
        
        # Initialize ROS node
        rospy.init_node('three_view_capture', anonymous=True)
        
        # Subscribe to image topics
        rospy.Subscriber(self.front_topic, Image, self.img_front_callback, queue_size=10)
        rospy.Subscriber(self.left_topic, Image, self.img_left_callback, queue_size=10)
        rospy.Subscriber(self.right_topic, Image, self.img_right_callback, queue_size=10)
        
        # print(f"Subscribed topics:")
        # print(f"  Front view: {self.front_topic}")
        # print(f"  Left view: {self.left_topic}")
        # print(f"  Right view: {self.right_topic}")
        # print("Waiting for image data...")
    
    def img_front_callback(self, msg):
        """Front-view camera callback."""
        if len(self.img_front_deque) >= 10:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)
    
    def img_left_callback(self, msg):
        """Left-view camera callback."""
        if len(self.img_left_deque) >= 10:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)
    
    def img_right_callback(self, msg):
        """Right-view camera callback."""
        if len(self.img_right_deque) >= 10:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)
    
    def get_synchronized_frames(self):
        """Get synchronized three-view images."""
        # Check that enough data is available
        if len(self.img_front_deque) == 0 or len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0:
            return None
        
        # Get latest timestamps
        front_time = self.img_front_deque[-1].header.stamp.to_sec()
        left_time = self.img_left_deque[-1].header.stamp.to_sec()
        right_time = self.img_right_deque[-1].header.stamp.to_sec()
        
        # Check timestamps are close enough (within 0.1s)
        max_time_diff = max(abs(front_time - left_time), 
                           abs(front_time - right_time), 
                           abs(left_time - right_time))
        
        if max_time_diff > 0.1:  # treat as synced within 100ms
            print(f"Images not synchronized: time diff {max_time_diff:.3f} s")
            return None
        
        # Get images
        front_msg = self.img_front_deque[-1]
        left_msg = self.img_left_deque[-1]
        right_msg = self.img_right_deque[-1]
        
        # Convert to OpenCV format
        try:
            front_img = self.bridge.imgmsg_to_cv2(front_msg, 'bgr8')
            left_img = self.bridge.imgmsg_to_cv2(left_msg, 'bgr8')
            right_img = self.bridge.imgmsg_to_cv2(right_msg, 'bgr8')
        except Exception as e:
            print(f"Image conversion failed: {e}")
            return None
        
        return front_img, left_img, right_img
    
    def save_images(self, front_img, left_img, right_img, output_dir='.'):
        """Save three-view images."""
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save images
        front_path = os.path.join(output_dir, 'front_view.jpg')
        left_path = os.path.join(output_dir, 'left_view.jpg')
        right_path = os.path.join(output_dir, 'right_view.jpg')
        
        cv2.imwrite(front_path, front_img)
        cv2.imwrite(left_path, left_img)
        cv2.imwrite(right_path, right_img)
        
        print("Images saved:")
        print(f"  Front view: {front_path}")
        print(f"  Left view: {left_path}")
        print(f"  Right view: {right_path}")
        
        return front_path, left_path, right_path
    
    def capture_once(self, output_dir='.'):
        """Capture three-view images once."""
        print("Starting three-view image capture...")
        
        # Wait for image data
        for i in range(50):  # wait up to 5 seconds
            frames = self.get_synchronized_frames()
            if frames is not None:
                front_img, left_img, right_img = frames
                # print(f"Got synchronized images (attempt {i+1})")
                # print(f"  Front view size: {front_img.shape}")
                # print(f"  Left view size: {left_img.shape}")
                # print(f"  Right view size: {right_img.shape}")
                
                # Save images
                return self.save_images(front_img, left_img, right_img, output_dir)
            
            rospy.sleep(0.1)  # wait 0.1s
        
        print("Unable to get synchronized image data")
        return None
    
    def run(self, output_dir='.'):
        """Main run loop."""
        try:
            # Capture images once
            result = self.capture_once(output_dir)
            
            if result:
                # print("\nCapture complete!")
                return True
            else:
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

    # Create capturer
    capture = ThreeViewImageCapture()

    # Capture images
    success = capture.run(output_dir)
    
    if success:
        pass
        # print("\nImage capture succeeded!")
        # print(f"Images saved under: {output_dir}")
    else:
        print("\nImage capture failed!")
        print("Please check:")
        print("  1. ROS environment is set up (source /opt/ros/noetic/setup.bash)")
        print("  2. Camera node is started (roslaunch realsense2_camera multi_camera.launch)")
        print("  3. Camera topics are publishing correctly")
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())