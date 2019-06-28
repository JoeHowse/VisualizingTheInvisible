import cv2

try:
    from PySpinCapture import PySpinCapture
except ImportError:
    PySpinCapture = None


__author__ = 'Joseph Howse'
__copyright__ = 'Copyright (c) 2018, Nummist Media Corporation Limited'
__credits__ = ['Joseph Howse']
__license__ = 'BSD 3-Clause'
__version__ = '0.0.1'
__maintainer__ = 'Joseph Howse'
__email__ = 'josephhowse@nummist.com'
__status__ = 'Prototype'


def main():

    if PySpinCapture is not None:
        capture = PySpinCapture(0, roi=(0, 0, 1920, 1200), binning_radius=1,
                                is_monochrome=True)
    else:
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    success, image = capture.read()
    while success:
        cv2.imshow('Live Video', image)
        if cv2.waitKey(1) != -1:
            # The user pressed a key.
            # Quit.
            break
        success, image = capture.read()


if __name__ == '__main__':
    main()
