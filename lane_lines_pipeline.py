import cv2
import matplotlib.pyplot as plt
import calibration
import perspective
import gradients

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


    # all further tests are on test1 that has a curve
    test_image = cv2.imread("test_images/test1.jpg")
    undistorted = calibration.undistort(test_image)
    cv2.imwrite("output_images/test1_undistort.jpg", undistorted)

    # color_masked = gradients.filter_colors_hsv(undistorted)
    # cv2.imwrite("output_images/test1_color_masked.jpg", color_masked)

    # testing perspective transform
    warped = perspective.warp(undistorted)
    cv2.imwrite("output_images/test1_perspective.jpg", warped)

    grad = gradients.apply_gradient(warped)

    # plt.imshow(grad)
    # plt.show()

    # binary = gradients.apply_gradient(warped, 3)
    # plt.imshow(binary)
    # plt.show()


    # print(binary.shape)
    #
    # to_save = cv2.cvtColor(binary,cv2.COLOR_GRAY2BGR)
    #
    # print(to_save.shape)
    # cv2.imwrite("output_images/test1_binary.jpg", binary)

