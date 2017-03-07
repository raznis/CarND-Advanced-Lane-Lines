import cv2

import calibration
import perspective

if __name__=="__main__":

    calibration = calibration.Calibration()

    perspective = perspective.Perspective()

    #performing calibration
    calibration.calibrate()
    # testing calibration
    test_image = cv2.imread("test_images/straight_lines1.jpg")
    undistorted = calibration.undistort(test_image)
    cv2.imwrite("output_images/straight_lines1_undistort.jpg", undistorted)

    # testing perspective transform
    warped = perspective.warp(undistorted)
    cv2.imwrite("output_images/straight_lines1_perspective.jpg", warped)

    test_image = cv2.imread("test_images/test1.jpg")
    undistorted = calibration.undistort(test_image)
    cv2.imwrite("output_images/test1_undistort.jpg", undistorted)

    # testing perspective transform
    warped = perspective.warp(undistorted)
    cv2.imwrite("output_images/test1_perspective.jpg", warped)