# cvutilities

Miscellaneous helper functions for fetching and processing OpenPose data and camera calibration data

All of these functions need to be reorganized and grouped with similar functionality from other packages/repos.

## Installation

* Clone this repo to any convenient location on your local drive

* From within your Python development environment (base environment or a virtual environment), run `pip install -e LOCAL_PATH_TO_REPO`

* `pip` will install the `cvutilities` package in your Python development environment (along with all dependencies)

* You can then import the modules like any other Python package (e.g., `import cvutilities.openpose_utilities`)

## Testing

Test script is in `tests/3d_reconstruction_example.py`

To run this script (and to use any of the functions that access camera/OpenPose data from S3), you need to set the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables to keys belonging to an account that is a member of the Wildflower `cameras` group.
