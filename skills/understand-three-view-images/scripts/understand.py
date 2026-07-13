#!/usr/bin/env python3
"""Offline / multi-image VLM helper (OpenAI-compatible Ark endpoint)."""

from __future__ import annotations

import argparse
import base64
import os

from openai import OpenAI

API_KEY = os.environ.get("ARK_API_KEY") or os.environ.get("VOLCENGINE_API_KEY") or ""
BASE_URL = os.environ.get(
    "ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"
)
ARK_MODEL = os.environ.get("ARK_MODEL", "doubao-seed-1-8-251228")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SKILL_DIR = os.path.dirname(_SCRIPT_DIR)
_DEFAULT_TMP = os.path.join(_SKILL_DIR, "tmp", "front_preview.jpg")

client = None
if API_KEY:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_images(image_paths, prompt):
    if client is None:
        raise RuntimeError("ARK_API_KEY / VOLCENGINE_API_KEY is not set")
    content_list = [{"type": "text", "text": prompt}]
    for image_path in image_paths:
        base64_image = encode_image(image_path)
        content_list.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
        )
    response = client.chat.completions.create(
        model=ARK_MODEL,
        messages=[{"role": "user", "content": content_list}],
        max_tokens=1024,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze multiple images with a prompt")
    parser.add_argument("--prompt", type=str, default="", help="Prompt text for the model")
    parser.add_argument(
        "--image",
        action="append",
        default=None,
        help="Image path (repeatable). Default: skill tmp/front_preview.jpg",
    )
    args = parser.parse_args()

    image_paths = args.image or [os.environ.get("THREE_VIEW_TMP_IMAGE", _DEFAULT_TMP)]
    system_prompt = "Images are captured by cameras on the robot. Answer briefly based on the image content:"
    user_prompt = args.prompt.strip() if args.prompt.strip() else "What is on the table?"
    prompt = system_prompt + user_prompt

    if not API_KEY:
        print("Note: please set environment variable ARK_API_KEY (or VOLCENGINE_API_KEY).")
    else:
        missing_files = [p for p in image_paths if not os.path.exists(p)]
        if missing_files:
            print(f"Error: image file(s) not found: {missing_files}. Please fix the path and retry.")
        else:
            print(f"Requesting the VLM ({len(image_paths)} images), please wait...\n")
            try:
                print(analyze_images(image_paths, prompt))
            except Exception as e:
                print(f"Request failed, error: {e}")
