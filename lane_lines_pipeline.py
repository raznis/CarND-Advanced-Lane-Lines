import cv2

import calibration

if __name__=="__main__":

    calibration = calibration.Calibration()

    #performing calibration
    calibration.calibrate()

    # testing calibration
    test_image = cv2.imread("camera_cal/calibration2.jpg")
    undistorted = calibration.undistort(test_image)
    cv2.imwrite("output_images/calibration2_undistort.jpg", undistorted)
