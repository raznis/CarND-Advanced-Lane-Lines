import cv2
import numpy as np


def normalized(img):
    return np.uint8(255 * img / np.max(np.absolute(img)))


def to_RGB(img):
    if img.ndim == 2:
        img_normalized = normalized(img)
        return np.dstack((img_normalized, img_normalized, img_normalized))
    elif img.ndim == 3:
        return img
    else:
        return None


def compose_diagScreen(curverad=0, offset=0, mainDiagScreen=None,
                       diag1=None, diag2=None, diag3=None, diag4=None, diag5=None, diag6=None, diag7=None,
                       diag8=None, diag9=None):
    # middle panel text example
    # using cv2 for drawing text in diagnostic pipeline.
    font = cv2.FONT_HERSHEY_COMPLEX
    middlepanel = np.zeros((120, 1280, 3), dtype=np.uint8)
    cv2.putText(middlepanel, 'Estimated lane curve radius: {0:.1f}'.format(curverad), (30, 60), font, 1, (255, 0, 0), 2)
    cv2.putText(middlepanel, 'Estimated Meters from lane center: {0:.1f}'.format(offset), (30, 90), font, 1, (255, 0, 0),
                2)

    # assemble the screen example
    diagScreen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    if mainDiagScreen is not None:
        diagScreen[0:720, 0:1280] = mainDiagScreen
    if diag1 is not None:
        diagScreen[0:240, 1280:1600] = cv2.resize(to_RGB(diag1), (320, 240), interpolation=cv2.INTER_AREA)
    if diag2 is not None:
        diagScreen[0:240, 1600:1920] = cv2.resize(to_RGB(diag2), (320, 240), interpolation=cv2.INTER_AREA)
    if diag3 is not None:
        diagScreen[240:480, 1280:1600] = cv2.resize(to_RGB(diag3), (320, 240), interpolation=cv2.INTER_AREA)
    if diag4 is not None:
        diagScreen[240:480, 1600:1920] = cv2.resize(to_RGB(diag4), (320, 240), interpolation=cv2.INTER_AREA) * 4
    if diag7 is not None:
        diagScreen[600:1080, 1280:1920] = cv2.resize(to_RGB(diag7), (640, 480), interpolation=cv2.INTER_AREA) * 4
    diagScreen[720:840, 0:1280] = middlepanel
    if diag5 is not None:
        diagScreen[840:1080, 0:320] = cv2.resize(to_RGB(diag5), (320, 240), interpolation=cv2.INTER_AREA)
    if diag6 is not None:
        diagScreen[840:1080, 320:640] = cv2.resize(to_RGB(diag6), (320, 240), interpolation=cv2.INTER_AREA)
    if diag9 is not None:
        diagScreen[840:1080, 640:960] = cv2.resize(to_RGB(diag9), (320, 240), interpolation=cv2.INTER_AREA)
    if diag8 is not None:
        diagScreen[840:1080, 960:1280] = cv2.resize(to_RGB(diag8), (320, 240), interpolation=cv2.INTER_AREA)

    return diagScreen