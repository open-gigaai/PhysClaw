#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
capture_and_understand_one_view.py
Capture front-view RGB and send it to a VLM (Volcengine Ark).
"""

from __future__ import annotations

import argparse
import base64
import os
import sys

import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from volcenginesdkarkruntime import Ark

API_KEY = os.environ.get("ARK_API_KEY") or os.environ.get("VOLCENGINE_API_KEY") or ""
BASE_URL = os.environ.get(
    "ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"
)
ARK_MODEL = os.environ.get("ARK_MODEL", "doubao-seed-2-0-mini-260428")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SKILL_DIR = os.path.dirname(_SCRIPT_DIR)
TMP_IMAGE = os.environ.get(
    "THREE_VIEW_TMP_IMAGE",
    os.path.join(_SKILL_DIR, "tmp", "front_preview.jpg"),
)

FRONT_TOPIC = "/camera_f/color/image_raw"

client = None
if API_KEY:
    client = Ark(
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=30,
        max_retries=1,
    )


def capture_front_image(timeout_s=2.0):
    """Wait for one front-view frame and return an OpenCV image."""
    bridge = CvBridge()
    try:
        msg = rospy.wait_for_message(FRONT_TOPIC, Image, timeout=timeout_s)
    except rospy.ROSException:
        print("Unable to get front view image data")
        return None

    try:
        return bridge.imgmsg_to_cv2(msg, "bgr8")
    except Exception as e:
        print(f"Image conversion failed: {e}")
        return None


def encode_cv_image_to_base64(cv_img):
    ok, buf = cv2.imencode(".jpg", cv_img)
    if not ok:
        print("Image encoding failed")
        return None
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def extract_response_text(response):
    texts = []
    for item in response.output:
        if getattr(item, "type", None) != "message":
            continue
        for content in item.content:
            if getattr(content, "type", None) == "output_text":
                texts.append(content.text)
    return "\n".join(texts)


def analyze_image(base64_image, prompt):
    if client is None:
        raise RuntimeError("ARK_API_KEY / VOLCENGINE_API_KEY is not set")
    response = client.responses.create(
        model=ARK_MODEL,
        thinking={"type": "disabled"},
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}"},
                    {"type": "input_text", "text": prompt},
                ],
            }
        ],
    )
    return extract_response_text(response)


def main():
    parser = argparse.ArgumentParser(description="Capture front view and analyze with a VLM")
    parser.add_argument("--prompt", type=str, default="", help="Prompt text for the model")
    parser.add_argument("--timeout", type=float, default=5.0, help="Timeout seconds for waiting a frame")
    args = parser.parse_args()

    rospy.init_node("capture_and_understand_one_view", anonymous=True)

    system_prompt = "Images are captured by cameras on the robot. Answer briefly based on the image content:"
    user_prompt = args.prompt.strip() if args.prompt.strip() else "Please describe what is in the image"
    prompt = system_prompt + user_prompt

    if not API_KEY:
        print("Note: please set environment variable ARK_API_KEY (or VOLCENGINE_API_KEY).")
        return 1

    cv_img = capture_front_image(timeout_s=args.timeout)
    if cv_img is None:
        return 1

    os.makedirs(os.path.dirname(TMP_IMAGE), exist_ok=True)
    cv2.imwrite(TMP_IMAGE, cv_img)
    print(f"Temporary image saved: {TMP_IMAGE}")

    base64_image = encode_cv_image_to_base64(cv_img)
    if base64_image is None:
        return 1
    print("Requesting the VLM (1 image), please wait...\n")
    try:
        answer = analyze_image(base64_image, prompt)
        print(answer)
        return 0
    except Exception as e:
        print(f"Request failed, error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
