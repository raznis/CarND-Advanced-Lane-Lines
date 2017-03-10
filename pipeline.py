import cv2
import matplotlib.pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip

import calibration
import lane_finder
import perspective
import gradients
import drawer
import visualization

import numpy as np

class pipeline:
    def __init__(self):
        self.left_fit = None
        self.right_fit = None
        self.calibration = calibration.Calibration()
        self.perspective = perspective.Perspective()
        # performing calibration
        self.calibration.calibrate()

    def process_video(self):
        # video = 'harder_challenge_video'
        # video = 'challenge_video'
        video = 'project_video'
        white_output = '{}_done_2.mp4'.format(video)
        clip1 = VideoFileClip('{}.mp4'.format(video)).subclip(38, 44)
        white_clip = clip1.fl_image(self.process_image)  # NOTE: this function expects color images!!
        white_clip.write_videofile(white_output, audio=False)

    def process_image(self, image):
        undistorted = self.calibration.undistort(image)
        mask = gradients.apply_gradient(undistorted)#, separate_channels=False)
        warped = np.uint8(self.perspective.warp(mask))
        # if self.left_fit is None or self.right_fit is None:
            #need to compute from scratch
        curr_left_fit, curr_right_fit, histogram = lane_finder.first_time_lane_find(warped)
        # else:
        #     self.left_fit, self.right_fit = lane_finder.second_time_lane_find(warped, self.left_fit, self.right_fit)
        if self.left_fit is None:
            self.left_fit = curr_left_fit
            self.right_fit = curr_right_fit

        diff_left = np.absolute(curr_left_fit - self.left_fit)
        diff_right = np.absolute(curr_right_fit - self.right_fit)
        # print("Diff left: " + str(diff_left))
        # print("Diff right: " + str(diff_right))

        left_curverad, right_curverad, distance_from_center = lane_finder.compute_radius(curr_left_fit, curr_right_fit)

        if curr_left_fit[0] * curr_right_fit[0] > 0.0 or (left_curverad / right_curverad < 3.0 and right_curverad / left_curverad < 3.0):
            self.left_fit = self.left_fit * 0.8 + curr_left_fit * 0.2
            self.right_fit = self.right_fit * 0.8 + curr_right_fit * 0.2

        # print(self.left_fit[0], self.right_fit[0], left_curverad, right_curverad)

        result = drawer.draw(undistorted, self.left_fit, self.right_fit, self.perspective, int((left_curverad + right_curverad) / 2), distance_from_center)
        return visualization.compose_diagScreen(int((left_curverad + right_curverad) / 2), distance_from_center, result, undistorted, mask, warped)


if __name__ == "__main__":
    pipe = pipeline()
    pipe.process_video()


