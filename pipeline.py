import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip

import calibration
import lane_finder
import perspective
import gradients
import drawer
from visualization import compose_diagScreen
from visualization import to_RGB
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
        white_output = '{}_test_1.mp4'.format(video)
        clip1 = VideoFileClip('{}.mp4'.format(video)).subclip(31, 44)
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
            if len(self.left_fits) < 20:
                curr_left_fit, curr_right_fit, histogram = lane_finder.first_time_lane_find(warped)
            else:
                curr_left_fit, curr_right_fit = lane_finder.second_time_lane_find(warped, self.left_fits[-1], self.right_fits[-1])
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

        smoothing_param = 13    # 13
        history_weight = 1.0    # 1.0
        if len(self.left_fits) <= smoothing_param:
            left_fit = self.left_fits[-1]
            right_fit = self.right_fits[-1]
        else:
            left_fit_window = self.left_fits[-smoothing_param:-1]   #make -2 to exclude last fit
            left_fit_window_sorted = sorted(left_fit_window, key=lambda x: x[0])
            right_fit_window = self.right_fits[-smoothing_param:-1] #make -2 to exclude last fit
            right_fit_window_sorted = sorted(right_fit_window, key=lambda x: x[0])
            left_fit = np.average(left_fit_window_sorted[int(smoothing_param*0.25):int(smoothing_param*0.75)+1],axis=0) * history_weight + self.left_fits[-1] * (1-history_weight)
            right_fit = np.average(right_fit_window_sorted[int(smoothing_param*0.25):int(smoothing_param*0.75)+1],axis=0) * history_weight + self.right_fits[-1] * (1-history_weight)

        result = drawer.draw(undistorted, left_fit, right_fit, self.perspective, self.radius, distance_from_center)
        visualization = compose_diagScreen(self.radius, distance_from_center, result, undistorted, filtered, mask, warped)

        if self.total_frames == 4:
            cv2.imwrite("output_images/example_undistorted.jpg", cv2.cvtColor(undistorted, cv2.COLOR_RGB2BGR))
            cv2.imwrite("output_images/example_filtered.jpg", cv2.cvtColor(filtered, cv2.COLOR_RGB2BGR))
            cv2.imwrite("output_images/example_mask.jpg", to_RGB(mask))
            cv2.imwrite("output_images/example_warped.jpg", to_RGB(warped))
            cv2.imwrite("output_images/example_result.jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            cv2.imwrite("output_images/example_visualization.jpg", cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

        return visualization



if __name__ == "__main__":
    pipe = pipeline()
    #process the video
    skipped, total = pipe.process_video()
    print("skipped frames: " + str(skipped) + " out of " + str(total))


