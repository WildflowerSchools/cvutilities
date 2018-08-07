import openpose_helper_functions
import camera_helper_functions
import networkx as nx
import numpy as np
import json


classroom_name = 'camera-sandbox'
datetime = np.datetime64('2018-07-04T18:23:00')
camera_names = ['camera01', 'camera02', 'camera03', 'camera04']

camera_calibration_data_all_cameras = camera_helper_functions.fetch_camera_calibration_data_from_local_drive_multiple_cameras(
    camera_names)

openpose_data_all_cameras = openpose_helper_functions.fetch_openpose_data_from_s3_multiple_cameras(
    classroom_name,
    datetime,
    camera_names)

matched_poses_3d, matched_projection_errors, match_indices = openpose_helper_functions.calculate_matched_poses_3d_multiple_cameras(
    openpose_data_all_cameras,
    camera_calibration_data_all_cameras)

print('\nMatched poses:')
print(matched_poses_3d)

print('\nProjection errors:')
print(matched_projection_errors)

print('\nMatch indices:')
print(match_indices)
