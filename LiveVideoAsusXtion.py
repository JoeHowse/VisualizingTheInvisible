import cv2
import numpy


__author__ = 'Joseph Howse'
__copyright__ = 'Copyright (c) 2019, Nummist Media Corporation Limited'
__credits__ = ['Joseph Howse']
__license__ = 'BSD 3-Clause'
__version__ = '0.0.1'
__maintainer__ = 'Joseph Howse'
__email__ = 'josephhowse@nummist.com'
__status__ = 'Prototype'


def main():

    capture = cv2.VideoCapture(cv2.CAP_OPENNI2_ASUS)
    channel = cv2.CAP_OPENNI_IR_IMAGE

    success = capture.grab()
    if success:
        success, image = capture.retrieve(flag=channel)
    while success:
        if image is not None:
            if channel == cv2.CAP_OPENNI_IR_IMAGE:
                # Assume the image is 10-bit.
                # Convert it to 8-bit.
                image = (image >> 2).astype(numpy.uint8)
            elif channel == cv2.CAP_OPENNI_DEPTH_MAP:
                # Assume the image is 12-bit (max depth 4.096m).
                # Convert it to 8-bit.
                image = (image >> 4).astype(numpy.uint8)
            cv2.imshow('Live Video', image)
        keycode = cv2.waitKey(1)
        if keycode == ord('1'):
            channel = cv2.CAP_OPENNI_IR_IMAGE
        elif keycode == ord('2'):
            channel = cv2.CAP_OPENNI_DEPTH_MAP
        elif keycode == ord('3'):
            channel = cv2.CAP_OPENNI_VALID_DEPTH_MASK
        elif keycode == ord('4'):
            channel = cv2.CAP_OPENNI_DISPARITY_MAP
        elif keycode == ord('5'):
            channel = cv2.CAP_OPENNI_POINT_CLOUD_MAP
        elif keycode == 27:
            # The user pressed the escape key.
            # Quit.
            break
        success = capture.grab()
        if success:
            success, image = capture.retrieve(flag=channel)


if __name__ == '__main__':
    main()
