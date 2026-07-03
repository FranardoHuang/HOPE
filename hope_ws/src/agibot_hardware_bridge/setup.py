from setuptools import find_packages, setup

package_name = "agibot_hardware_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/agibot_bridge.launch.py"]),
        (f"share/{package_name}/config", ["config/agibot_bridge.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="HOPE",
    maintainer_email="redspry@gmail.com",
    description="Bridge between WBC runner and AGI A3 body-drive topics",
    license="MIT",
    entry_points={
        "console_scripts": [
            "agibot_hardware_bridge = agibot_hardware_bridge.bridge_node:main",
        ],
    },
)
