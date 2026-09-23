#!/usr/bin/env python3

import argparse
import ctypes
import time

import numpy as np
import cv2
from PIL import Image, ImageDraw
import tensorrt as trt


INPUT_H = 544
INPUT_W = 960

CLASS_NAMES = {
    1: "crop",
    2: "weed",
}


class CudaRuntime:
    MEMCPY_HOST_TO_DEVICE = 1
    MEMCPY_DEVICE_TO_HOST = 2

    def __init__(self, path="/cuda-libs/libcudart.so.13"):
        self.lib = ctypes.CDLL(path)

        self.lib.cudaMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
        ]
        self.lib.cudaMalloc.restype = ctypes.c_int

        self.lib.cudaFree.argtypes = [
            ctypes.c_void_p
        ]
        self.lib.cudaFree.restype = ctypes.c_int

        self.lib.cudaHostAlloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        self.lib.cudaHostAlloc.restype = ctypes.c_int

        self.lib.cudaFreeHost.argtypes = [
            ctypes.c_void_p
        ]
        self.lib.cudaFreeHost.restype = ctypes.c_int

        self.lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.lib.cudaMemcpyAsync.restype = ctypes.c_int

        self.lib.cudaStreamCreate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p)
        ]
        self.lib.cudaStreamCreate.restype = ctypes.c_int

        self.lib.cudaStreamSynchronize.argtypes = [
            ctypes.c_void_p
        ]
        self.lib.cudaStreamSynchronize.restype = ctypes.c_int

        self.lib.cudaStreamDestroy.argtypes = [
            ctypes.c_void_p
        ]
        self.lib.cudaStreamDestroy.restype = ctypes.c_int

        self.lib.cudaGetErrorString.argtypes = [
            ctypes.c_int
        ]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p

    def check(self, code, operation):
        if code != 0:
            msg = self.lib.cudaGetErrorString(code).decode()
            raise RuntimeError(
                f"{operation} failed: CUDA error {code}: {msg}"
            )

    def malloc(self, size):
        ptr = ctypes.c_void_p()

        self.check(
            self.lib.cudaMalloc(
                ctypes.byref(ptr),
                size,
            ),
            "cudaMalloc",
        )

        return ptr

    def free(self, ptr):
        if ptr:
            self.check(
                self.lib.cudaFree(ptr),
                "cudaFree",
            )

    def host_alloc(self, shape, dtype):
        dtype = np.dtype(dtype)

        count = int(np.prod(shape))
        size = count * dtype.itemsize

        ptr = ctypes.c_void_p()

        self.check(
            self.lib.cudaHostAlloc(
                ctypes.byref(ptr),
                size,
                0,
            ),
            "cudaHostAlloc",
        )

        ctype = np.ctypeslib.as_ctypes_type(dtype)

        backing_type = ctype * count
        backing = backing_type.from_address(ptr.value)

        array = np.ctypeslib.as_array(
            backing
        ).reshape(shape)

        return array, ptr, backing

    def free_host(self, ptr):
        if ptr:
            self.check(
                self.lib.cudaFreeHost(ptr),
                "cudaFreeHost",
            )

    def memcpy_htod_async(
        self,
        device_ptr,
        array,
        stream,
    ):
        self.check(
            self.lib.cudaMemcpyAsync(
                device_ptr,
                ctypes.c_void_p(array.ctypes.data),
                array.nbytes,
                self.MEMCPY_HOST_TO_DEVICE,
                stream,
            ),
            "cudaMemcpyAsync H2D",
        )

    def memcpy_dtoh_async(
        self,
        array,
        device_ptr,
        stream,
    ):
        self.check(
            self.lib.cudaMemcpyAsync(
                ctypes.c_void_p(array.ctypes.data),
                device_ptr,
                array.nbytes,
                self.MEMCPY_DEVICE_TO_HOST,
                stream,
            ),
            "cudaMemcpyAsync D2H",
        )

    def create_stream(self):
        stream = ctypes.c_void_p()

        self.check(
            self.lib.cudaStreamCreate(
                ctypes.byref(stream)
            ),
            "cudaStreamCreate",
        )

        return stream

    def synchronize(self, stream):
        self.check(
            self.lib.cudaStreamSynchronize(stream),
            "cudaStreamSynchronize",
        )

    def destroy_stream(self, stream):
        self.check(
            self.lib.cudaStreamDestroy(stream),
            "cudaStreamDestroy",
        )


def preprocess(image):
    original_w, original_h = image.size

    # PIL image is already RGB.
    rgb = np.asarray(image.convert("RGB"))

    tensor = cv2.dnn.blobFromImage(
        rgb,
        scalefactor=1.0 / 255.0,
        size=(INPUT_W, INPUT_H),
        mean=(0.0, 0.0, 0.0),
        swapRB=False,
        crop=False,
    )

    return np.ascontiguousarray(tensor), (original_w, original_h)


def sigmoid(x):
    positive = x >= 0

    result = np.empty_like(x, dtype=np.float32)

    result[positive] = 1.0 / (
        1.0 + np.exp(-x[positive])
    )

    exp_x = np.exp(x[~positive])
    result[~positive] = exp_x / (1.0 + exp_x)

    return result


def postprocess(
    pred_logits,
    pred_boxes,
    original_size,
    threshold,
):
    original_w, original_h = original_size

    logits = pred_logits[0].astype(np.float32)
    boxes = pred_boxes[0].astype(np.float32)

    scores = sigmoid(logits)

    num_classes = scores.shape[1]

    flat_scores = scores.reshape(-1)

    top_k = min(300, flat_scores.size)

    top_indices = np.argpartition(
        flat_scores,
        -top_k,
    )[-top_k:]

    top_indices = top_indices[
        np.argsort(flat_scores[top_indices])[::-1]
    ]

    detections = []

    for flat_idx in top_indices:
        score = float(flat_scores[flat_idx])

        if score < threshold:
            continue

        class_id = int(flat_idx % num_classes)
        query_id = int(flat_idx // num_classes)

        if class_id not in CLASS_NAMES:
            continue

        cx, cy, bw, bh = boxes[query_id]

        x1 = (cx - bw / 2.0) * original_w
        y1 = (cy - bh / 2.0) * original_h
        x2 = (cx + bw / 2.0) * original_w
        y2 = (cy + bh / 2.0) * original_h

        x1 = int(np.clip(x1, 0, original_w - 1))
        y1 = int(np.clip(y1, 0, original_h - 1))
        x2 = int(np.clip(x2, 0, original_w - 1))
        y2 = int(np.clip(y2, 0, original_h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append(
            {
                "class_id": class_id,
                "label": CLASS_NAMES[class_id],
                "score": score,
                "box": (x1, y1, x2, y2),
            }
        )

    return detections


def draw_detections(image, detections):
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)

    for det in detections:
        x1, y1, x2, y2 = det["box"]

        if det["class_id"] == 1:
            color = "green"
        else:
            color = "red"

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=color,
            width=3,
        )

        label = (
            f"{det['label']} "
            f"{det['score']:.2f}"
        )

        text_y = max(0, y1 - 15)

        draw.text(
            (x1, text_y),
            label,
            fill=color,
        )

    return image


class TRTModel:
    def __init__(self, engine_path):
        self.cuda = CudaRuntime()

        self.logger = trt.Logger(
            trt.Logger.WARNING
        )

        ctypes.CDLL(
            "/trt-libs/libnvinfer_plugin.so.10",
            mode=ctypes.RTLD_GLOBAL,
        )

        trt.init_libnvinfer_plugins(
            self.logger,
            "",
        )

        with open(engine_path, "rb") as f:
            self.runtime = trt.Runtime(
                self.logger
            )

            self.engine = (
                self.runtime.deserialize_cuda_engine(
                    f.read()
                )
            )

        if self.engine is None:
            raise RuntimeError(
                "Failed to deserialize TensorRT engine"
            )

        self.context = (
            self.engine.create_execution_context()
        )

        if self.context is None:
            raise RuntimeError(
                "Failed to create TensorRT context"
            )

        self.stream = self.cuda.create_stream()

        self.host_buffers = {}
        self.host_allocations = {}
        self.device_buffers = {}

        print("TensorRT I/O tensors:")

        for i in range(
            self.engine.num_io_tensors
        ):
            name = self.engine.get_tensor_name(i)

            shape = tuple(
                self.context.get_tensor_shape(
                    name
                )
            )

            if any(dim < 0 for dim in shape):
                raise RuntimeError(
                    f"Dynamic shape unresolved: "
                    f"{name} {shape}"
                )

            dtype = np.dtype(
                trt.nptype(
                    self.engine.get_tensor_dtype(
                        name
                    )
                )
            )

            # Pinned/page-locked host memory
            host, host_ptr, backing = (
                self.cuda.host_alloc(
                    shape,
                    dtype,
                )
            )

            device = self.cuda.malloc(
                host.nbytes
            )

            self.host_buffers[name] = host

            # Keep both pointer and ctypes backing alive.
            self.host_allocations[name] = (
                host_ptr,
                backing,
            )

            self.device_buffers[name] = (
                device
            )

            ok = (
                self.context.set_tensor_address(
                    name,
                    int(device.value),
                )
            )

            if not ok:
                raise RuntimeError(
                    f"Could not set address "
                    f"for {name}"
                )

            mode = (
                self.engine.get_tensor_mode(
                    name
                )
            )

            print(
                f"  {name}: "
                f"shape={shape}, "
                f"dtype={dtype}, "
                f"mode={mode}"
            )

    def infer(self, tensor):
        input_name = None

        for i in range(
            self.engine.num_io_tensors
        ):
            name = self.engine.get_tensor_name(i)

            if (
                self.engine.get_tensor_mode(name)
                == trt.TensorIOMode.INPUT
            ):
                input_name = name
                break

        if input_name is None:
            raise RuntimeError(
                "TensorRT input tensor not found"
            )

        input_buffer = (
            self.host_buffers[input_name]
        )

        np.copyto(
            input_buffer,
            tensor.astype(
                input_buffer.dtype,
                copy=False,
            ),
        )

        start = time.perf_counter()

        # Async H2D on the TensorRT stream.
        self.cuda.memcpy_htod_async(
            self.device_buffers[input_name],
            input_buffer,
            self.stream,
        )

        # TensorRT executes after H2D because
        # everything is queued on the same stream.
        ok = self.context.execute_async_v3(
            stream_handle=int(
                self.stream.value
            )
        )

        if not ok:
            raise RuntimeError(
                "TensorRT inference failed"
            )

        # Queue D2H immediately after inference.
        for i in range(
            self.engine.num_io_tensors
        ):
            name = self.engine.get_tensor_name(i)

            if (
                self.engine.get_tensor_mode(name)
                != trt.TensorIOMode.OUTPUT
            ):
                continue

            self.cuda.memcpy_dtoh_async(
                self.host_buffers[name],
                self.device_buffers[name],
                self.stream,
            )

        # One synchronization for the entire
        # H2D -> inference -> D2H sequence.
        self.cuda.synchronize(
            self.stream
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        outputs = {}

        for i in range(
            self.engine.num_io_tensors
        ):
            name = self.engine.get_tensor_name(i)

            if (
                self.engine.get_tensor_mode(name)
                == trt.TensorIOMode.OUTPUT
            ):
                outputs[name] = (
                    self.host_buffers[name].copy()
                )

        return outputs, elapsed_ms

    def close(self):
        for ptr in (
            self.device_buffers.values()
        ):
            self.cuda.free(ptr)

        for host_ptr, _ in (
            self.host_allocations.values()
        ):
            self.cuda.free_host(
                host_ptr
            )

        self.cuda.destroy_stream(
            self.stream
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--engine",
        required=True,
    )

    parser.add_argument(
        "--image",
        required=True,
    )

    parser.add_argument(
        "--output",
        default="/output/result.jpg",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    args = parser.parse_args()

    original = Image.open(args.image)

    print(
        f"Original image: "
        f"{original.width}x{original.height}"
    )

    preprocess_start = time.perf_counter()

    tensor, original_size = preprocess(original)

    preprocess_ms = (
        time.perf_counter() - preprocess_start
    ) * 1000.0

    print(
        f"Network input: {tensor.shape}"
    )

    model = TRTModel(args.engine)

    try:
        outputs, inference_ms = model.infer(tensor)
    finally:
        model.close()

    if "pred_logits" not in outputs:
        raise RuntimeError(
            f"pred_logits not found: "
            f"{list(outputs.keys())}"
        )

    if "pred_boxes" not in outputs:
        raise RuntimeError(
            f"pred_boxes not found: "
            f"{list(outputs.keys())}"
        )

    print(
        "pred_logits:",
        outputs["pred_logits"].shape,
    )

    print(
        "pred_boxes:",
        outputs["pred_boxes"].shape,
    )

    post_start = time.perf_counter()

    detections = postprocess(
        outputs["pred_logits"],
        outputs["pred_boxes"],
        original_size,
        args.threshold,
    )

    result = draw_detections(
        original.copy(),
        detections,
    )

    postprocess_ms = (
        time.perf_counter() - post_start
    ) * 1000.0

    result.save(args.output, quality=95)

    pipeline_ms = (
        preprocess_ms
        + inference_ms
        + postprocess_ms
    )

    print()
    print(f"Preprocessing: {preprocess_ms:.2f} ms")
    print(
        f"TensorRT + transfers: "
        f"{inference_ms:.2f} ms"
    )
    print(
        f"Postprocessing: "
        f"{postprocess_ms:.2f} ms"
    )
    print(
        f"Pipeline total: "
        f"{pipeline_ms:.2f} ms"
    )

    print()
    print(
        f"Detections >= "
        f"{args.threshold:.2f}: "
        f"{len(detections)}"
    )

    for det in detections:
        print(
            f"  {det['label']:5s} "
            f"score={det['score']:.3f} "
            f"box={det['box']}"
        )

    print()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
