import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Calibration:
    def __init__(self):
        self._mtx = None
        self._dist = None

    def calibrate(self):

        print("Starting calibration...")

        # prepare object points
        nx = 9  # number of inside corners in x
        ny = 6  # number of inside corners in y

        obj_points = [] # 3d points in real world space
        img_points = [] # 2d points in image plane

        # prepare object points
        objp = np.zeros((nx*ny,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

        # Make a list of calibration images
        images = glob.glob("camera_cal/calibration*.jpg")
        for fname in images:
            img = cv2.imread(fname)
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            # If found, draw corners
            if ret == True:
                # Draw and display the corners
                if fname == "camera_cal/calibration2.jpg":
                    img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                    cv2.imwrite("output_images/calibration2_corners.jpg", img)

                img_points.append(corners) # adding image chessboard corners
                obj_points.append(objp) # adding the same grid points of a real chessboard

        #done processing all images
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img.shape[0:2], None, None)


        self._mtx = mtx
        self._dist = dist

        print("Done calibration.")

    def undistort(self, img):
        if self._mtx is None:
            self.calibrate()

        dst = cv2.undistort(img, self._mtx, self._dist, None, self._mtx)
        return dst
