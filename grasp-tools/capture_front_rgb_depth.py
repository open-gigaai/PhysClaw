#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Capture front camera RGB + depth-aligned depth and save as color.png / depth.png.

Topics default to TOPIC_FRONT_COLOR / TOPIC_FRONT_DEPTH from the environment
(configs/paths.env). Override for your camera drivers.
"""

import os
from collections import deque

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

RGB_TOPIC = os.environ.get("TOPIC_FRONT_COLOR", "/camera_f/color/image_raw")
DEPTH_TOPIC = os.environ.get("TOPIC_FRONT_DEPTH", "/camera_f/depth/image_raw")
EXPECTED_RGB_SHAPE = (480, 640, 3)
EXPECTED_DEPTH_SHAPE = (480, 640)


class FrontRgbDepthCapture:
    def __init__(self):
        self.bridge = CvBridge()
        self.rgb_deque = deque(maxlen=10)
        self.depth_deque = deque(maxlen=10)

        self.rgb_topic = RGB_TOPIC
        self.depth_topic = DEPTH_TOPIC

        rospy.init_node('front_rgb_depth_capture', anonymous=True)
        rospy.Subscriber(self.rgb_topic, Image, self._rgb_callback, queue_size=10)
        rospy.Subscriber(self.depth_topic, Image, self._depth_callback, queue_size=10)

    def _rgb_callback(self, msg):
        self.rgb_deque.append(msg)

    def _depth_callback(self, msg):
        self.depth_deque.append(msg)

    def get_synchronized_frames(self):
        if len(self.rgb_deque) == 0 or len(self.depth_deque) == 0:
            return None

        rgb_time = self.rgb_deque[-1].header.stamp.to_sec()
        depth_time = self.depth_deque[-1].header.stamp.to_sec()
        if abs(rgb_time - depth_time) > 0.1:
            print(f"Images not synchronized: time diff {abs(rgb_time - depth_time):.3f} s")
            return None

        try:
            rgb = self.bridge.imgmsg_to_cv2(self.rgb_deque[-1], 'bgr8')
            depth = self.bridge.imgmsg_to_cv2(self.depth_deque[-1], 'passthrough')
        except Exception as e:
            print(f"Image conversion failed: {e}")
            return None

        depth = self._normalize_depth(depth)
        depth = self._ensure_depth_shape(depth, rgb.shape[:2])
        return rgb, depth

    def _normalize_depth(self, depth_img):
        if depth_img.dtype == 'float32' or depth_img.dtype == 'float64':
            depth_mm = depth_img * 1000.0
            depth_mm[depth_mm < 0] = 0
            depth_mm[depth_mm > 65535] = 65535
            return depth_mm.astype('uint16')
        return depth_img

    def _ensure_depth_shape(self, depth, rgb_hw):
        """Match depth resolution to RGB. Prefer hardware D2C (480x640)."""
        rgb_h, rgb_w = rgb_hw
        depth_h, depth_w = depth.shape[:2]

        if (depth_h, depth_w) == (rgb_h, rgb_w):
            return depth

        if depth_h == 400 and depth_w == 640 and rgb_h == 480:
            # print(
            #     "Note: depth is 640x400; bottom-padded to align with RGB 480x640"
            # )
            return cv2.copyMakeBorder(depth, 0, 80, 0, 0, cv2.BORDER_CONSTANT, value=0)

        print(
            f"Warning: RGB={rgb_w}x{rgb_h}, depth={depth_w}x{depth_h}, "
            "resizing depth to RGB size."
        )
        return cv2.resize(depth, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)

    def save_images(self, rgb, depth, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        color_path = os.path.join(output_dir, 'color.png')
        depth_path = os.path.join(output_dir, 'depth.png')

        # print(f"RGB shape:   {rgb.shape}")
        # print(f"Depth shape: {depth.shape}")

        cv2.imwrite(color_path, rgb)
        cv2.imwrite(depth_path, depth)
        # print("Images saved:")
        # print(f"  RGB:   {color_path}")
        # print(f"  Depth: {depth_path}")
        return color_path, depth_path

    def capture_once(self, output_dir):
        # print("Starting front RGB + Depth capture...")
        # print(f"  RGB topic:   {self.rgb_topic}")
        # print(f"  Depth topic: {self.depth_topic}")
        for _ in range(50):
            frames = self.get_synchronized_frames()
            if frames is not None:
                rgb, depth = frames
                if rgb.shape != EXPECTED_RGB_SHAPE:
                    print(f"Warning: RGB size {rgb.shape} does not match expected {EXPECTED_RGB_SHAPE}")
                return self.save_images(rgb, depth, output_dir=output_dir)
            rospy.sleep(0.1)
        print("Unable to get synchronized image data")
        return None

    def run(self, output_dir):
        try:
            return self.capture_once(output_dir) is not None
        except rospy.ROSInterruptException:
            print("ROS interrupted")
            return False
        except Exception as e:
            print(f"Error during capture: {e}")
            return False


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'example_data')

    capture = FrontRgbDepthCapture()
    success = capture.run(output_dir)

    if not success:
        print("\nRGB+Depth image capture failed!")
        print("Please check:")
        print("  1. ROS environment is set up")
        print("  2. Your camera launch is running (set CAMERA_LAUNCH in configs/paths.env)")
        print(f"  3. Topics {RGB_TOPIC} and {DEPTH_TOPIC} are publishing")

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
