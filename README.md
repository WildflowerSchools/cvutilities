# cvutilities

Miscellaneous helper functions for fetching and processing OpenPose data and camera calibration data

All of these functions need to be reorganized and grouped with similar functionality from other packages/repos

To use the functions that access camera/OpenPose data from S3, you need to set the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables to keys belonging to an account that is a member of the Wildflower `cameras` group. 
