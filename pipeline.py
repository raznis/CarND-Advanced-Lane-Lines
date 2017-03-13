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
        self.left_fits = []
        self.right_fits = []
        self.calibration = calibration.Calibration()
        self.perspective = perspective.Perspective()
        # performing calibration
        self.calibration.calibrate()
        self.skipped_frames = 0
        self.total_frames = 0
        self.radius = -1.0


    def process_video(self):
        # video = 'harder_challenge_video'
        # video = 'challenge_video'
        video = 'project_video'
        white_output = '{}_done_2.mp4'.format(video)
        clip1 = VideoFileClip('{}.mp4'.format(video))#.subclip(35, 44)
        white_clip = clip1.fl_image(self.process_image)  # NOTE: this function expects color images!!
        white_clip.write_videofile(white_output, audio=False)
        return self.skipped_frames, self.total_frames

    def process_image(self, image):
        undistorted = self.calibration.undistort(image)
        filtered = gradients.filter_colors_hsv(undistorted)
        mask = gradients.get_edges(filtered, separate_channels=False)#, separate_channels=False)
        warped = np.uint8(self.perspective.warp(mask))
        # if self.left_fit is None or self.right_fit is None:
            #need to compute from scratch
        try:
            curr_left_fit, curr_right_fit, histogram = lane_finder.first_time_lane_find(warped)
        except TypeError:
            curr_left_fit, curr_right_fit = self.left_fits[-1], self.right_fits[-1]
            self.skipped_frames += 1

        left_curverad, right_curverad, distance_from_center = lane_finder.compute_radius(curr_left_fit, curr_right_fit)

        # if curr_left_fit[0] * curr_right_fit[0] > 0.0:
        self.left_fits.append(curr_left_fit)
        self.right_fits.append(curr_right_fit)
        # else:
        #     self.skipped_frames += 1
        self.total_frames += 1
        # print(self.left_fit[0], self.right_fit[0], left_curverad, right_curverad)
        curr_radius = int((left_curverad + right_curverad) / 2)
        if self.radius < 0:
            self.radius = curr_radius
        elif curr_radius / self.radius < 1.5:
            self.radius = self.radius * 0.95 + curr_radius * 0.05
        else:
            self.radius = self.radius * 0.95 + self.radius * 1.5 * 0.05

        smoothing_param = 20
        if len(self.left_fits) < smoothing_param:
            left_fit = self.left_fits[-1]
            right_fit = self.right_fits[-1]
        else:
            left_fit = [np.average(x) for x in zip(*self.left_fits[-smoothing_param:-1])]
            right_fit = [np.average(x) for x in zip(*self.right_fits[-smoothing_param:-1])]

        result = drawer.draw(undistorted, left_fit, right_fit, self.perspective, self.radius, distance_from_center)
        return visualization.compose_diagScreen(self.radius, distance_from_center, result, undistorted, filtered, mask, warped)


if __name__ == "__main__":
    pipe = pipeline()
    skipped, total = pipe.process_video()
    print("skipped frames: " + str(skipped) + " out of " + str(total))


