import cv2
import matplotlib.pyplot as plt
import calibration
import lane_finder
import perspective
import gradients
import numpy as np





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
    test_image = cv2.imread("test_images/test4.jpg")
    # test_image = cv2.cvtColor(test_image,cv2.COLOR_BGR2RGB)
    undistorted = calibration.undistort(test_image)
    cv2.imwrite("output_images/test4_undistorted.jpg", undistorted)
    mask = gradients.get_edges(undistorted, separate_channels=False)
    # plt.imshow(mask)
    # plt.show()

    warped = np.uint8(perspective.warp(mask))
    cv2.imwrite("output_images/test4_warped.jpg", warped)

    # plt.imshow(warped)
    # plt.show()

    left_fit, right_fit = lane_finder.first_time_lane_find(warped)

    y_eval = 719
    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
    print(left_curverad, right_curverad)



    # SECOND IMAGE:
    test_image = cv2.imread("test_images/test1.jpg")
    # test_image = cv2.cvtColor(test_image,cv2.COLOR_BGR2RGB)
    undistorted = calibration.undistort(test_image)
    cv2.imwrite("output_images/test1_undistorted.jpg", undistorted)
    mask = gradients.get_edges(undistorted, separate_channels=False)
    # plt.imshow(mask)
    # plt.show()
    warped = np.uint8(perspective.warp(mask))
    cv2.imwrite("output_images/test1_warped.jpg", warped)

    left_fit, right_fit = lane_finder.second_time_lane_find(warped, left_fit, right_fit)

    y_eval = 719
    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
    print(left_curverad, right_curverad)



    # f, axes = plt.subplots(2, 2, figsize=(24, 9))
    # f.tight_layout()
    # axes[0, 0].imshow(test_image)
    # axes[0, 0].set_title('original image', fontsize=50)
    # axes[1, 0].imshow(mask)
    # axes[1, 0].set_title('mask', fontsize=50)
    # axes[0, 1].imshow(warped)
    # axes[0, 1].set_title('warped', fontsize=50)
    # axes[1, 1].imshow(out_img)
    # axes[1, 1].plot(left_fitx, ploty, color='yellow')
    # axes[1, 1].plot(right_fitx, ploty, color='yellow')
    # axes[1, 1].set_title('lane lines', fontsize=50)
    # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    #
    # plt.show()

