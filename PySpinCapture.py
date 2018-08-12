import PySpin
import cv2


__author__ = 'Joseph Howse'
__copyright__ = 'Copyright (c) 2018, Nummist Media Corporation Limited'
__credits__ = ['Joseph Howse']
__license__ = 'BSD 3-Clause'
__version__ = '0.0.1'
__maintainer__ = 'Joseph Howse'
__email__ = 'josephhowse@nummist.com'
__status__ = 'Prototype'


class PySpinCapture:


    def __init__(self, index, roi, binning_radius=1, is_monochrome=False):

        self._system = system = PySpin.System.GetInstance()

        self._camera_list = system.GetCameras()

        self._camera = self._camera_list.GetByIndex(index)
        self._camera.Init()

        self._nodemap = self._camera.GetNodeMap()

        # Enable continuous acquisition mode.
        node_acquisition_mode = PySpin.CEnumerationPtr(self._nodemap.GetNode(
            'AcquisitionMode'))
        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName(
            'Continuous')
        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

        # Set the pixel format.
        node_pixel_format = PySpin.CEnumerationPtr(self._nodemap.GetNode('PixelFormat'))
        if is_monochrome:
            # Enable Mono8 mode.
            node_pixel_format_mono8 = PySpin.CEnumEntryPtr(
                node_pixel_format.GetEntryByName('Mono8'))
            pixel_format_mono8 = node_pixel_format_mono8.GetValue()
            node_pixel_format.SetIntValue(pixel_format_mono8)
        else:
            # Enable BGR8 mode.
            node_pixel_format_bgr8 = PySpin.CEnumEntryPtr(
                node_pixel_format.GetEntryByName('BGR8'))
            pixel_format_bgr8 = node_pixel_format_bgr8.GetValue()
            node_pixel_format.SetIntValue(pixel_format_bgr8)

        # Set the vertical binning radius.
        # The horizontal binning radius is automatically set to the same value.
        node_binning_vertical = PySpin.CIntegerPtr(self._nodemap.GetNode(
            'BinningVertical'))
        node_binning_vertical.SetValue(binning_radius)

        # Set the ROI.
        x, y, w, h  = roi
        node_offset_x = PySpin.CIntegerPtr(self._nodemap.GetNode('OffsetX'))
        node_offset_x.SetValue(x)
        node_offset_y = PySpin.CIntegerPtr(self._nodemap.GetNode('OffsetY'))
        node_offset_y.SetValue(y)
        node_width = PySpin.CIntegerPtr(self._nodemap.GetNode('Width'))
        node_width.SetValue(w)
        node_height = PySpin.CIntegerPtr(self._nodemap.GetNode('Height'))
        node_height.SetValue(h)

        self._camera.BeginAcquisition()


    def get(self, propId):
        if propId == cv2.CAP_PROP_FRAME_WIDTH:
            node_width = PySpin.CIntegerPtr(self._nodemap.GetNode('Width'))
            return float(node_width.GetValue())
        if propId == cv2.CAP_PROP_FRAME_HEIGHT:
            node_height = PySpin.CIntegerPtr(self._nodemap.GetNode('Height'))
            return float(node_height.GetValue())
        return 0.0


    def __del__(self):
        self.release()


    def read(self, image=None):

        camera_image = self._camera.GetNextImage()
        if camera_image.IsIncomplete():
            return False, None

        h = camera_image.GetHeight()
        w = camera_image.GetWidth()
        num_channels = camera_image.GetNumChannels()
        if num_channels > 1:
            camera_image_data = camera_image.GetData().reshape(h, w, num_channels)
        else:
            camera_image_data = camera_image.GetData().reshape(h, w)

        if image is None:
            image = camera_image_data.copy()
        else:
            image[:] = camera_image_data

        camera_image.Release()

        return True, image


    def release(self):

        self._camera.EndAcquisition()
        self._camera.DeInit()
        del self._camera

        self._camera_list.Clear()

        self._system.ReleaseInstance()
