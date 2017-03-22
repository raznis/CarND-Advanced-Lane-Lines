import cv2
import matplotlib.pyplot as plt
import calibration
import lane_finder
import perspective
import gradients
import drawer
import visualization
import numpy as np




if __name__=="__main__":

    calibration = calibration.Calibration()
    perspective = perspective.Perspective()

    #performing calibration
    calibration.calibrate()
    # testing calibration
    test_image = cv2.imread("test_images/straight_lines1.jpg")
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    undistorted = calibration.undistort(test_image)
    filtered = gradients.filter_colors_hsv(undistorted)
    mask = gradients.get_edges(filtered, separate_channels=False)  # , separate_channels=False)
    warped = np.uint8(perspective.warp(mask))

    cv2.imwrite("output_images/straight_lines1_undistorted.jpg", undistorted)
    cv2.imwrite("output_images/straight_lines1_filtered.jpg", filtered)
    cv2.imwrite("output_images/straight_lines1_mask.jpg", mask)
    cv2.imwrite("output_images/straight_lines1_warped.jpg", warped)

    curr_left_fit, curr_right_fit, histogram = lane_finder.first_time_lane_find(warped)
    left_curverad, right_curverad, distance_from_center = lane_finder.compute_radius(curr_left_fit, curr_right_fit)
    curr_radius = int((left_curverad + right_curverad) / 2)
    result = drawer.draw(undistorted, curr_left_fit, curr_right_fit, perspective, curr_radius, distance_from_center)

    cv2.imwrite("output_images/straight_lines1_result.jpg", result)
    visualization = visualization.compose_diagScreen(int((left_curverad +right_curverad) / 2), distance_from_center, result, undistorted, filtered, mask, warped)
    cv2.imwrite("output_images/straight_lines1_visualization.jpg", result)



