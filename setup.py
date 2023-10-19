from setuptools import find_packages, setup


def get_description():
    """Gets the description from the readme."""
    with open("README.md") as fh:
        long_description = ""
        header_count = 0
        for line in fh:
            if line.startswith("##"):
                header_count += 1
            if header_count < 2:
                long_description += line
            else:
                break
    return header_count, long_description


def get_version():
    """Gets the d4rl version."""
    path = "wrappers/__init__.py"
    with open(path) as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")

version = get_version()


setup(
    name="D4RL Wrapper",
    version=version,
    author="Gwangpyo Yoo",
    description="D4RL Wrapper for new gymnasium api",
    long_description_content_type="text/markdown",
    keywords=["Reinforcement Learning", "Datasets", "D4RL", "Wrapper"],
    python_requires=">=3.8",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "d4rl_gymnasium_wrappers": [
            "wrappers/*",
        ]
    },
    install_requires=[
        "gym",
        "gymnasium",
        "numpy",
        "mujoco_py",
        "pybullet",
        "h5py",
        "termcolor",  # adept_envs dependency
        "click",  # adept_envs dependency
        "dm_control>=1.0.3",
        "mjrl @ git+https://github.com/aravindr93/mjrl@master#egg=mjrl",
    ],
)