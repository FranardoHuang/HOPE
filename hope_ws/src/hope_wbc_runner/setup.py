import os
from glob import glob

from setuptools import find_packages, setup

package_name = "hope_wbc_runner"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="HOPE Maintainers",
    maintainer_email="maintainer@example.com",
    description="Staged, safety-gated WBC runner for model_15200 (bring-up companion to planner_imitate).",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "wbc_runner_node = hope_wbc_runner.wbc_runner:main",
        ],
    },
)
