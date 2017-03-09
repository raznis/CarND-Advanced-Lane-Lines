import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Perspective:
    def __init__(self):
        self._M = None
        self._Minv = None

    def warp(self, img):
        if self._M is None:
            print("Computing perspective transform matrix...")
            # src = np.float32(
            #     [[233, 700],    # bottom left
            #      [1070, 700],   # bottom right
            #      [702, 461],    # top right       711, 486
            #      [580, 461]]    # top left         570, 486
            # )
            #
            # dst = np.float32(
            #     [[280, 690],  # bottom left
            #      [1000, 690],  # bottom right
            #      [1000, 30],  # top right
            #      [280, 30]]  # top right
            # )

            (h, w) = (img.shape[0], img.shape[1])
            # Define source points
            src = np.float32([[w // 2 - 76, h * .625], [w // 2 + 76, h * .625], [-100, h], [w + 100, h]])
            # Define corresponding destination points
            dst = np.float32([[100, 0], [w - 100, 0], [100, h], [w - 100, h]])

            self._M = cv2.getPerspectiveTransform(src, dst)

            self._Minv = cv2.getPerspectiveTransform(dst, src)

            print("Done perspective transform matrix.")

        img_size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img,self._M, img_size, flags=cv2.INTER_LINEAR)

    def unwarp(self, img):
        img_size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, self._Minv, img_size, flags=cv2.INTER_LINEAR)