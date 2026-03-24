import os
from glob import glob

from setuptools import find_packages, setup

package_name = "rviz_mesh_publisher"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "meshes"), glob("meshes/*.*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="yl43338-admin",
    maintainer_email="lanbugart@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "publisher = rviz_mesh_publisher.publisher:main",
        ],
    },
)
