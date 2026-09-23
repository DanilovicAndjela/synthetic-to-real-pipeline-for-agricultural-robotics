from setuptools import setup

package_name = "rtdetr_ros"

setup(
    name=package_name,
    version="0.0.1",

    packages=[package_name],

    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
    ],

    install_requires=["setuptools"],
    zip_safe=True,

    maintainer="RT-DETR Project",
    maintainer_email="dev@example.com",

    description=(
        "ROS 2 RT-DETR TensorRT perception pipeline"
    ),

    license="MIT",

    entry_points={
        "console_scripts": [
            (
                "video_publisher = "
                "rtdetr_ros.video_publisher:main"
            ),
            (
                "inference_node = "
                "rtdetr_ros.inference_node:main"
            ),
        ],
    },
)
