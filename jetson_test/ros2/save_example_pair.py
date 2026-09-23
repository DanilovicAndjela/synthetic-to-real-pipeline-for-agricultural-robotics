import os
from collections import OrderedDict

import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)
from sensor_msgs.msg import CompressedImage


INPUT_TOPIC = "/camera/image/compressed"
OUTPUT_TOPIC = "/rtdetr/annotated_image/compressed"

OUT_DIR = "/output/final_example"

# Прескочи првих 100 обрађених кадрова,
# па сачувај први наредни пар.
SKIP_OUTPUT_FRAMES = 100


class PairSaver(Node):

    def __init__(self):
        super().__init__("final_example_pair_saver")

        os.makedirs(OUT_DIR, exist_ok=True)

        self.input_cache = OrderedDict()
        self.output_count = 0
        self.saved = False

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(
            CompressedImage,
            INPUT_TOPIC,
            self.input_callback,
            qos,
        )

        self.create_subscription(
            CompressedImage,
            OUTPUT_TOPIC,
            self.output_callback,
            qos,
        )

        self.get_logger().info(
            "Waiting for matching input/output frame..."
        )

    @staticmethod
    def key(msg):
        return (
            msg.header.stamp.sec,
            msg.header.stamp.nanosec,
        )

    @staticmethod
    def decode(msg):
        data = np.frombuffer(msg.data, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    def input_callback(self, msg):

        image = self.decode(msg)

        if image is None:
            return

        key = self.key(msg)
        self.input_cache[key] = image

        # Не треба нам бесконачан cache.
        while len(self.input_cache) > 300:
            self.input_cache.popitem(last=False)

    def output_callback(self, msg):

        if self.saved:
            return

        self.output_count += 1

        if self.output_count <= SKIP_OUTPUT_FRAMES:
            return

        key = self.key(msg)

        if key not in self.input_cache:
            return

        annotated = self.decode(msg)

        if annotated is None:
            return

        original = self.input_cache[key]

        input_path = os.path.join(
            OUT_DIR,
            "jetson_input_frame.jpg",
        )

        output_path = os.path.join(
            OUT_DIR,
            "jetson_detections_frame.jpg",
        )

        cv2.imwrite(
            input_path,
            original,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )

        cv2.imwrite(
            output_path,
            annotated,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )

        self.saved = True

        self.get_logger().info(
            "===== FRAME PAIR SAVED ====="
        )
        self.get_logger().info(input_path)
        self.get_logger().info(output_path)
        self.get_logger().info(
            f"Matched timestamp: {key[0]}.{key[1]:09d}"
        )
        self.get_logger().info(
            "============================"
        )


def main():

    rclpy.init()

    node = PairSaver()

    while rclpy.ok() and not node.saved:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
