"""SAM3 text-prompt segmentation: save binary mask to data_dir/mask.png."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image

SAM3_CHECKPOINT = os.environ.get(
    "SAM3_CHECKPOINT",
    os.path.join(
        os.path.expanduser("~"),
        "models",
        "huggingface",
        "models--facebook--sam3",
        "sam3.pt",
    ),
)
SAM3_CONFIDENCE_THRESHOLD = 0.3
DEFAULT_MASK_NAME = 'mask.png'
DEFAULT_SOCKET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.sam3_seg.sock')


class Sam3Segmenter:
    """Load SAM3 once and reuse for repeated text-prompt segmentation."""

    def __init__(
        self,
        checkpoint_path: str = SAM3_CHECKPOINT,
        confidence_threshold: float = SAM3_CONFIDENCE_THRESHOLD,
        resolution: int = 1008,
        compile_model: bool = False,
    ) -> None:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        t0 = time.time()
        self.model = build_sam3_image_model(
            load_from_HF=False,
            checkpoint_path=checkpoint_path,
            compile=compile_model,
        )
        self.processor = Sam3Processor(
            self.model,
            resolution=resolution,
            confidence_threshold=confidence_threshold,
        )
        print(f'SAM3 model ready in {time.time() - t0:.2f}s', flush=True)

    def segment_from_path(self, image_path: str, text_prompt: str) -> np.ndarray:
        image = Image.open(image_path).convert('RGB')
        return self.segment_from_image(image, text_prompt)

    def segment_from_image(self, image: Image.Image, text_prompt: str) -> np.ndarray:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            inference_state = self.processor.set_image(image)
            output = self.processor.set_text_prompt(state=inference_state, prompt=text_prompt)

        masks = output['masks']
        scores = output['scores']
        if masks.numel() == 0 or len(scores) == 0:
            raise ValueError(
                f'SAM3 found no objects for prompt={text_prompt!r} '
                f'(confidence_threshold={self.processor.confidence_threshold})'
            )

        best_idx = int(scores.argmax().item())
        mask_np = masks[best_idx].squeeze().detach().cpu().numpy().astype(bool)
        score = float(scores[best_idx].item())
        n_detections = len(scores)
        n_fg = int(np.count_nonzero(mask_np))
        h, w = mask_np.shape
        print(
            f'SAM3 prompt={text_prompt!r}: picked detection {best_idx}/{n_detections - 1} '
            f'(score={score:.3f}), {n_fg}/{h * w} foreground pixels '
            f'({100.0 * n_fg / (h * w):.1f}%)',
            flush=True,
        )
        if n_fg == 0:
            raise ValueError(f'SAM3 mask has no foreground pixels for prompt={text_prompt!r}')
        return mask_np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='SAM3 text-prompt segmentation')
    parser.add_argument(
        '--text_prompt',
        help='Text prompt for SAM3 to segment the target object',
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./example_data',
        help='Directory containing color.png; mask.png is written here',
    )
    parser.add_argument(
        '--serve',
        action='store_true',
        help='Keep SAM3 loaded and serve segmentation requests on a Unix socket',
    )
    parser.add_argument(
        '--via-server',
        action='store_true',
        help='Send segmentation request to a running SAM3 server instead of loading the model',
    )
    parser.add_argument(
        '--socket',
        type=str,
        default=DEFAULT_SOCKET_PATH,
        help=f'Unix socket path for --serve / --via-server (default: {DEFAULT_SOCKET_PATH})',
    )
    parser.add_argument(
        '--compile',
        action='store_true',
        help='Enable torch.compile for SAM3 (slow first run, may help repeated inference)',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress logs (pipeline uses this by default)',
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def save_binary_mask(mask: np.ndarray, mask_path: str, *, quiet: bool = False) -> None:
    """Save bool mask as BW PNG: white=foreground, black=background."""
    Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_path)
    if not quiet:
        print(f'Saved mask to {mask_path}', flush=True)


def _handle_request(segmenter: Sam3Segmenter, request: dict) -> dict:
    color_path = request.get('color_path')
    text_prompt = request.get('text_prompt')
    mask_path = request.get('mask_path')
    if not color_path or not text_prompt or not mask_path:
        return {'ok': False, 'error': 'color_path, text_prompt, and mask_path are required'}

    color_path = os.path.abspath(color_path)
    mask_path = os.path.abspath(mask_path)
    if not os.path.isfile(color_path):
        return {'ok': False, 'error': f'color image not found: {color_path}'}

    t0 = time.time()
    try:
        mask = segmenter.segment_from_path(color_path, text_prompt)
        save_binary_mask(mask, mask_path)
    except Exception as exc:  # noqa: BLE001 - return error to client
        return {'ok': False, 'error': str(exc)}

    return {
        'ok': True,
        'mask_path': mask_path,
        'elapsed_s': round(time.time() - t0, 3),
    }


def _recv_message(conn: socket.socket) -> dict:
    chunks: list[bytes] = []
    while True:
        chunk = conn.recv(4096)
        if not chunk:
            raise ConnectionError('client disconnected before sending a full request')
        chunks.append(chunk)
        if chunk.endswith(b'\n'):
            break
    return json.loads(b''.join(chunks).decode('utf-8'))


def _send_message(conn: socket.socket, payload: dict) -> None:
    conn.sendall((json.dumps(payload) + '\n').encode('utf-8'))


def serve_forever(socket_path: str, compile_model: bool = False) -> None:
    socket_path = os.path.abspath(socket_path)
    if os.path.exists(socket_path):
        os.unlink(socket_path)

    segmenter = Sam3Segmenter(compile_model=compile_model)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    os.chmod(socket_path, 0o666)
    server.listen(5)
    print(f'SAM3 server listening on {socket_path}', flush=True)

    try:
        while True:
            conn, _ = server.accept()
            with conn:
                try:
                    request = _recv_message(conn)
                    response = _handle_request(segmenter, request)
                    _send_message(conn, response)
                except (BrokenPipeError, ConnectionError, ConnectionResetError):
                    pass
                except Exception as exc:  # noqa: BLE001 - keep server alive
                    try:
                        _send_message(conn, {'ok': False, 'error': str(exc)})
                    except (BrokenPipeError, ConnectionError, ConnectionResetError):
                        pass
    finally:
        server.close()
        if os.path.exists(socket_path):
            os.unlink(socket_path)


def request_via_server(
    socket_path: str,
    color_path: str,
    text_prompt: str,
    mask_path: str,
    timeout_s: float = 120.0,
    *,
    quiet: bool = False,
) -> None:
    socket_path = os.path.abspath(socket_path)
    if not os.path.exists(socket_path):
        raise FileNotFoundError(
            f'SAM3 server socket not found: {socket_path}. '
            'Start one with: python run_seg.py --serve'
        )

    payload = {
        'color_path': os.path.abspath(color_path),
        'text_prompt': text_prompt,
        'mask_path': os.path.abspath(mask_path),
    }
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.settimeout(timeout_s)
    try:
        conn.connect(socket_path)
    except ConnectionRefusedError as exc:
        raise ConnectionRefusedError(
            f'SAM3 server socket exists but is not accepting connections: {socket_path}. '
            'The server process was likely killed; restart with: bash start_seg_server.sh'
        ) from exc
    try:
        _send_message(conn, payload)
        response = _recv_message(conn)
    finally:
        conn.close()

    if not response.get('ok'):
        raise RuntimeError(response.get('error', 'SAM3 server request failed'))

    elapsed_s = response.get('elapsed_s')
    if elapsed_s is not None:
        if quiet:
            print(f'SAM3: {text_prompt!r} ({elapsed_s:.3f}s)', flush=True)
        else:
            print(f'SAM3 server inference finished in {elapsed_s:.3f}s', flush=True)


def run_segmentation(
    text_prompt: str,
    data_dir: str,
    via_server: bool = False,
    socket_path: str = DEFAULT_SOCKET_PATH,
    compile_model: bool = False,
    *,
    quiet: bool = False,
) -> None:
    color_path = os.path.join(data_dir, 'color.png')
    mask_path = os.path.join(data_dir, DEFAULT_MASK_NAME)

    if via_server:
        request_via_server(socket_path, color_path, text_prompt, mask_path, quiet=quiet)
        return

    segmenter = Sam3Segmenter(compile_model=compile_model)
    mask = segmenter.segment_from_path(color_path, text_prompt)
    save_binary_mask(mask, mask_path, quiet=quiet)


def main() -> None:
    cfgs = parse_args()
    if cfgs.serve:
        serve_forever(cfgs.socket, compile_model=cfgs.compile)
        return

    if not cfgs.text_prompt:
        print('Error: --text_prompt is required unless --serve is used', file=sys.stderr)
        sys.exit(2)

    run_segmentation(
        text_prompt=cfgs.text_prompt,
        data_dir=cfgs.data_dir,
        via_server=cfgs.via_server,
        socket_path=cfgs.socket,
        compile_model=cfgs.compile,
        quiet=cfgs.quiet,
    )


if __name__ == '__main__':
    main()
