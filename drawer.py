import cv2
import numpy as np


def draw(img, left_fit, right_fit, perspective, curve_radius, distance_from_center):
    color_warp = np.zeros_like(img).astype(np.uint8)

    fity = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0] * fity ** 2 + left_fit[1] * fity + left_fit[2]
    right_fitx = right_fit[0] * fity ** 2 + right_fit[1] * fity + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, fity]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, fity])))])
    pts = np.hstack((pts_left, pts_right))
    pts = np.array(pts, dtype=np.int32)

    cv2.fillPoly(color_warp, pts, (0, 255, 0))

    newwarp = perspective.unwarp(color_warp)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Curve Radius: ' + str(curve_radius), (300, 100), font, 2, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, 'Distance From Center: ' + "{:2.2f}".format(distance_from_center), (250, 200), font, 2, (255, 0, 0), 2, cv2.LINE_AA)

    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    return result
