import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define a function that applies Sobel x or y,
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Apply the following steps to img
    # Convert to grayscale
    # Take the absolute value of the derivative or gradient
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return this mask as your binary_output image

    return binary_output


# Define a function that applies Sobel x and y,
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale not necessary (already single channel)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Take the absolute value of the derivative or gradient
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * gradmag / np.max(gradmag))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return binary_output

# Define a function that applies Sobel x and y,
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Convert to grayscale not necessary (already single channel)
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output


# Define a function that thresholds the S-channel of HLS
def hls_select(img, thresh=(90, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output


def color_thresh(image, threshold=(0, 255)):
    mask = np.zeros_like(image)
    mask[(image > threshold[0]) & (image <= threshold[1])] = 1
    return mask


def filter_colors_hsv(img):
    """
    Convert image to HSV color space and suppress any colors
    outside of the defined color ranges
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    yellow_dark = np.array([15, 127, 127], dtype=np.uint8)
    yellow_light = np.array([25, 255, 255], dtype=np.uint8)
    yellow_range = cv2.inRange(img, yellow_dark, yellow_light)

    white_dark = np.array([0, 0, 200], dtype=np.uint8)
    white_light = np.array([255, 30, 255], dtype=np.uint8)
    white_range = cv2.inRange(img, white_dark, white_light)
    yellows_or_whites = yellow_range | white_range
    img = cv2.bitwise_and(img, img, mask=yellows_or_whites)

    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

def apply_gradient(image, ksize=7):
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), orient='x', sobel_kernel=ksize, thresh=(5, 100))
    grady = abs_sobel_thresh(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), orient='y', sobel_kernel=ksize, thresh=(5, 100))
    mag_binary = mag_thresh(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), sobel_kernel=ksize, mag_thresh=(50, 150))
    dir_binary = dir_threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), sobel_kernel=ksize, thresh=(0.7, 1.3))

    # Convert to HLS colorspace
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS).astype(np.float)
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    gradx_s_channel = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=7, thresh=(50, 255))

    combined = np.zeros_like(gradx)
    # combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    combined[((gradx == 1) & (grady == 1) & (gradx_s_channel == 1)) | ((dir_binary == 1) & (gradx_s_channel == 1))] = 1

    # f, axes = plt.subplots(2, 2, figsize=(24, 9))
    # f.tight_layout()
    # axes[0, 0].imshow(image)
    # axes[0, 0].set_title('image', fontsize=50)
    # axes[0, 1].imshow(l_channel)
    # axes[0, 1].set_title('l_channel', fontsize=50)
    # axes[1, 0].imshow(gradx_s_channel)
    # axes[1, 0].set_title('gradx_s_channel', fontsize=50)
    # axes[1, 1].imshow(s_channel)
    # axes[1, 1].set_title('s_channel', fontsize=50)
    # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    #
    # plt.show()


    return combined


def get_edges(image, separate_channels=False):
    # Convert to HLS color space and separate required channel
    hls = cv2.cvtColor(np.copy(image), cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:, :, 2]
    # Get a combination of all gradient thresholding masks
    gradient_x = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=3, thresh=(20, 100))
    gradient_y = abs_sobel_thresh(s_channel, orient='y', sobel_kernel=3, thresh=(20, 100))
    magnitude = mag_thresh(s_channel, sobel_kernel=3, mag_thresh=(20, 100))
    direction = dir_threshold(s_channel, sobel_kernel=3, thresh=(0.7, 1.3))
    gradient_mask = np.zeros_like(s_channel)
    gradient_mask[((gradient_x == 1) & (gradient_y == 1)) | ((magnitude == 1) & (direction == 1))] = 1
    # Get a color thresholding mask
    color_mask = color_thresh(s_channel, threshold=(170, 255))

    if separate_channels:
        return np.dstack((np.zeros_like(s_channel), gradient_mask, color_mask))
    else:
        mask = np.zeros_like(gradient_mask)
        mask[(gradient_mask == 1) | (color_mask == 1)] = 1
    return mask
