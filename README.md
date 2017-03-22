##Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./output_images/calibration2_corners.jpg "corners"
[image1]: ./output_images/calibration2_undistorted.jpg "Undistorted Chessboard"
[image2]: ./output_images/example_undistorted.jpg "Undistorted Road"
[image3]: ./output_images/example_filtered.jpg "Color Filter"
[image4]: ./output_images/example_mask.jpg "Final Binary masking"
[image5]: ./output_images/example_warped.jpg "Warped Binary Image"
[image6]: ./output_images/example_result.jpg "With Lane Lines"
[image7]: ./output_images/sliding_window.png "Sliding Window Lane Find"
[image8]: ./output_images/second_level_lane_find.png "Second Level Lane Find"
[image9]: ./output_images/example_visualization.jpg "Debugging all stages"
[video1]: ./project_video_done.mp4 "Video"


Camera Calibration
---


The code for this step is contained in the calibration class  (`calibration.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. An example of the detection of corners can be seem here:

![alt text][image0]  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the same chessboard image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

Pipeline (single images)
---
1. Provide an example of a distortion-corrected image.

To demonstrate this step, here is an example of a road image after distortion correction:
![alt text][image2]

2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 110 through 126, and at lines 165 to 184 in `gradients.py`). I first applied a color mask for yellow and white in the HSV color space, and then then applied sobel, magnitude and direction thresholding on the S channel. Pixels where turned on if they passed one of the two masks. Here's an example of my output for this step. First, color thresholding:

![alt text][image3]

And then the final binary masking:

![alt text][image4]

3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in lines 13 through 30 in the file `perspective.py`.  The `warp()` function takes as inputs an image (`img`). I chose the hardcode the source and destination points in the following manner:

```
(h, w) = (img.shape[0], img.shape[1])
# Define source points
src = np.float32([[w // 2 - 76, h * .625], [w // 2 + 76, h * .625], [-100, h], [w + 100, h]])
# Define corresponding destination points
dst = np.float32([[100, 0], [w - 100, 0], [100, h], [w - 100, h]])

```
This resulted in the following source and destination points:
| Source        | Destination   | 
|:-------------:|:-------------:| 
| 564     450     | 100		0        | 
| 716     450     | 1180	0      |
| -100    720     | 100   	720     |
| 1380   720     | 1180   	720       |

I verified that my perspective transform was working as expected by checking multiple test images and their respective warped images. An example of such an image can be seen here:

![alt text][image5]

4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Code for this part can be found in `lane_finder.py`, lines 9 to 145. First, given the warped binary image, I computed a histogram of "lit" pixels w.r.t. the x-axis. The two peaks of the histogram correspond to the two lane locations. Then, using sliding windows, I found the locations of the most lit pixels. These rectangles provided the scope of the points to be used when computing the 2nd order polynomial fit of the lanes. An example is below:

![alt text][image7]

After we have found the lanes lines for the first time, we can use this information to restrict the computation and save computation time. There is no need to compute the histogram, and we use our previousely found lanes as the scope for computing the polynomial fit. A visualization of this is below:

![alt text][image8]

In order to smooth out outlier frames, the output for each frame represented the average of the last 13 frames, removing the top and bottom 25%, w.r.t. the coefficient of x^2 of the polyinomial fit of the lane.


5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 153 through 181 in my code in `lane_finder.py`

6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 5 through 28 in my code in `drawer.py` in the function `draw()`.  Here is an example of my result on a test image:

![alt text][image6]

In addition, in order to debug all the stages of the pipeline, I implemented a visualizer to display all the stages in a single image. I also utilized this in computing the output video. Here's an example of this:
![alt text][image9]

---

###Pipeline (video)
---
1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_done.mp4)

---

Discussion
---
1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The first thing I noticed is that this (computer vision) approach requires a lot of fine tuning of the parameters. What appears to work well on some test images, can break down on very similar images, as were included in the video. Looking at potential areas of imprevement, I believe further tuning of the thresholding would create a more robust pipeline. I used masking of yellow and white in my pipeline, and this might fail when lanes have different colors, or when lighting conditions are completely different.

A great deal of time was also spent on smoothing the lane fits, in order to disregard outliers and avoid catastrophic detections. This would probably be less necessary if the thresholding was more robust.

