#!/usr/bin/env python


import datetime
import math
import os
import threading
import timeit

import cv2
import numpy
import wx

try:
    from PySpinCapture import PySpinCapture
except ImportError:
    PySpinCapture = None


WX_MAJOR_VERSION = int(wx.__version__.split('.')[0])


__author__ = 'Joseph Howse'
__copyright__ = 'Copyright (c) 2018, Nummist Media Corporation Limited'
__credits__ = ['Joseph Howse']
__license__ = 'BSD 3-Clause'
__version__ = '0.0.1'
__maintainer__ = 'Joseph Howse'
__email__ = 'josephhowse@nummist.com'
__status__ = 'Prototype'


FLOAT_TYPE = numpy.float64

FLANN_INDEX_LSH = 6

MAP_TO_PLANE = 0
MAP_TO_CUBOID = 1
MAP_TO_CYLINDER = 2


def convert_to_gray(src, dst=None):
    weight = 1.0 / 3.0
    return cv2.transform(src, numpy.array([[weight, weight, weight]], FLOAT_TYPE), dst)


def map_point_onto_plane(point_2D, image_size, image_scale):
    x, y = point_2D
    w, h = image_size
    return (image_scale * (x - 0.5 * w), image_scale * (y - 0.5 * h), 0.0)


def map_point_onto_cuboid(point_2D, image_size, image_scale):

    x, y = point_2D
    w, h = image_size

    y_3D = image_scale * (y - 0.5 * h)
    w_3D = image_scale * w / 8.0

    segment_x1 = w * 0.25
    if x < segment_x1:
        # Map the point onto the cuboid's front face.
        return (image_scale * x - w_3D, y_3D, -w_3D)

    segment_x2 = w * 0.5
    if x < segment_x2:
        # Map the point onto the cuboid's left face.
        segment_w = segment_x2 - segment_x1
        return (w_3D, y_3D, image_scale * (x - segment_x1) - w_3D)

    segment_x3 = w * 0.75
    if x < segment_x3:
        # Map the point onto the cuboid's back face.
        segment_w = segment_x3 - segment_x2
        return (w_3D - image_scale * (x - segment_x2), y_3D, w_3D)

    # Map the point onto the cuboid's right face.
    segment_w = w - segment_x3
    return (-w_3D, y_3D, w_3D - image_scale * (x - segment_x3))


def map_point_onto_cylinder(point_2D, image_size, image_scale):
    x, y = point_2D
    w, h = image_size
    image_real_radius = image_scale * w / (2.0 * math.pi)
    theta = 2.0 * math.pi * ((x / float(w)) - 0.25)
    return (image_real_radius * math.cos(theta),
            image_scale * (y - 0.5 * h),
            image_real_radius * math.sin(theta))


def map_points_to_3D(points_2D, image_size, image_real_height, mapping_type):

    w, h = image_size
    image_scale = image_real_height / h

    if mapping_type is MAP_TO_CUBOID:
        mapping_function = map_point_onto_cuboid
    elif mapping_type is MAP_TO_CYLINDER:
        mapping_function = map_point_onto_cylinder
    else:  # MAP_TO_PLANE
        mapping_function = map_point_onto_plane

    points_3D = [mapping_function(point_2D, image_size, image_scale)
                 for point_2D in points_2D]
    return numpy.array(points_3D, FLOAT_TYPE)


def map_vertices_to_3D(image_size, image_real_height, mapping_type):

    w, h = image_size

    if mapping_type is not MAP_TO_PLANE:
        if mapping_type is MAP_TO_CUBOID:
            num_segments = 4
        else:  # MAP_TO_CYLINDER
            num_segments = 8
        segment_indices = list(range(num_segments))
        num_vertices = 2 * num_segments
        xs = [w * i / float(num_segments) for i in segment_indices]
        vertices_2D = [(x, 0) for x in xs] + [(x, h) for x in xs[::-1]]
        vertex_indices_by_face = [segment_indices]  # Top
        for i in range(num_segments - 1):  # Sides
            vertex_indices_by_face += [[i, i+1, num_vertices-i-2, num_vertices-i-1]]
        vertex_indices_by_face += [[  # Last side
            num_segments-1, 0, num_vertices-1, num_segments]]
        vertex_indices_by_face += [list(range(num_segments, 2 * num_segments))]  # Bottom
    else:  # MAP_TO_PLANE
        vertices_2D = [(0, 0), (w, 0), (w, h), (0, h)]
        vertex_indices_by_face = [[0, 1, 2, 3]]

    vertices_3D = map_points_to_3D(vertices_2D, image_size, image_real_height,
                                   mapping_type)
    return vertices_3D, vertex_indices_by_face


class VisualizingTheInvisible(wx.Frame):


    def __init__(self, capture, is_monochrome=False, diagonal_fov_degrees=70.0,
                 target_fps=25.0, reference_image_path='reference_image.jpg',
                 reference_image_real_height=1.0, reference_image_mapping=MAP_TO_PLANE,
                 saved_scenes_path='saved_scenes', title='Visualizing the Invisible'):

        self._capture = capture
        success, trial_image = capture.read()
        if success:
            # Use the actual image dimensions.
            h, w = trial_image.shape[:2]
            is_monochrome = (len(trial_image.shape) == 2)
        else:
            # Use the nominal image dimensions.
            w = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self._image_size = (w, h)
        self._is_monochrome = is_monochrome

        diagonal_image_size = (w ** 2.0 + h ** 2.0) ** 0.5
        diagonal_fov_radians = diagonal_fov_degrees * math.pi / 180.0
        focal_length = 0.5 * diagonal_image_size / math.tan(0.5 * diagonal_fov_radians)
        self._camera_matrix = numpy.array(
            [[focal_length,          0.0, 0.5 * w],
             [         0.0, focal_length, 0.5 * h],
             [         0.0,          0.0,     1.0]], FLOAT_TYPE)

        self._distortion_coefficients = None

        self._rotation_vector = None
        self._translation_vector = None

        self._kalman = cv2.KalmanFilter(18, 6)

        self._kalman.processNoiseCov = numpy.identity(18, FLOAT_TYPE) * 1e-5
        self._kalman.measurementNoiseCov = numpy.identity(6, FLOAT_TYPE) * 1e-2
        self._kalman.errorCovPost = numpy.identity(18, FLOAT_TYPE)

        self._kalman.measurementMatrix = numpy.array(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            FLOAT_TYPE)

        self._init_kalman_transition_matrix(target_fps)

        self._was_tracking = False

        self._reference_image_real_height = reference_image_real_height
        reference_axis_length = 0.5 * reference_image_real_height

        #-----------------------------------------------------------------------------
        # BEWARE!
        #-----------------------------------------------------------------------------
        #
        # OpenCV's coordinate system has non-standard axis directions:
        #   +X:  object's left; viewer's right from frontal view
        #   +Y:  down
        #   +Z:  object's backward; viewer's forward from frontal view
        #
        # Negate them all to convert to right-handed coordinate system (like OpenGL):
        #   +X:  object's right; viewer's left from frontal view
        #   +Y:  up
        #   +Z:  object's forward; viewer's backward from frontal view
        #
        #-----------------------------------------------------------------------------
        self._reference_axis_points_3D = numpy.array(
            [[                   0.0,                    0.0,                    0.0],
             [-reference_axis_length,                    0.0,                    0.0],
             [                   0.0, -reference_axis_length,                    0.0],
             [                   0.0,                    0.0, -reference_axis_length]],
            FLOAT_TYPE)

        self._bgr_image = None
        self._rgb_image = None
        self._gray_image = None
        self._mask = None

        self._rgb_image_front_buffer = None
        self._rgb_image_front_buffer_lock = threading.Lock()

        # Create and configure the feature detector.
        patchSize = 31
        self._feature_detector = cv2.ORB_create(nfeatures=250, scaleFactor=1.2,
                                                nlevels=16, edgeThreshold=patchSize,
                                                patchSize=patchSize)

        bgr_reference_image = cv2.imread(reference_image_path, cv2.IMREAD_COLOR)
        reference_image_h, reference_image_w = bgr_reference_image.shape[:2]
        reference_image_resize_factor = (2.0 * h) / reference_image_h
        bgr_reference_image = cv2.resize(
            bgr_reference_image, (0, 0), None, reference_image_resize_factor,
            reference_image_resize_factor, cv2.INTER_CUBIC)
        gray_reference_image = convert_to_gray(bgr_reference_image)
        reference_mask = numpy.empty_like(gray_reference_image)

        # Find keypoints and descriptors for multiple segments of the reference image.
        reference_keypoints = []
        self._reference_descriptors = numpy.empty((0, 32), numpy.uint8)
        num_segments_y = 6
        num_segments_x = 6
        for segment_y, segment_x in numpy.ndindex((num_segments_y, num_segments_x)):
            y0 = reference_image_h * segment_y // num_segments_y - patchSize
            x0 = reference_image_w * segment_x // num_segments_x - patchSize
            y1 = reference_image_h * (segment_y + 1) // num_segments_y + patchSize
            x1 = reference_image_w * (segment_x + 1) // num_segments_x + patchSize
            reference_mask.fill(0)
            cv2.rectangle(reference_mask, (x0, y0), (x1, y1), 255, cv2.FILLED)
            more_reference_keypoints, more_reference_descriptors = \
                self._feature_detector.detectAndCompute(gray_reference_image,
                                                        reference_mask)
            if more_reference_descriptors is None:
                # No keypoints were found for this segment.
                continue
            reference_keypoints += more_reference_keypoints
            self._reference_descriptors = numpy.vstack(
                (self._reference_descriptors, more_reference_descriptors))

        cv2.drawKeypoints(gray_reference_image, reference_keypoints,
                          bgr_reference_image,
                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        ext_i = reference_image_path.rfind('.')
        reference_image_keypoints_path = \
            reference_image_path[:ext_i] + '_keypoints' + reference_image_path[ext_i:]
        cv2.imwrite(reference_image_keypoints_path, bgr_reference_image)

        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12,
                            multi_probe_level=1)
        search_params = dict(checks=50)
        self._descriptor_matcher = cv2.FlannBasedMatcher(index_params, search_params)
        self._descriptor_matcher.add([self._reference_descriptors])

        reference_points_2D = [keypoint.pt for keypoint in reference_keypoints]
        self._reference_points_3D = map_points_to_3D(
            reference_points_2D, gray_reference_image.shape[::-1],
            reference_image_real_height, reference_image_mapping)

        self._reference_vertices_3D, self._reference_vertex_indices_by_face = \
            map_vertices_to_3D(gray_reference_image.shape[::-1],
                               reference_image_real_height, reference_image_mapping)

        self._saved_scenes_path = saved_scenes_path

        style = wx.CLOSE_BOX | wx.MINIMIZE_BOX | wx.CAPTION | wx.SYSTEM_MENU | \
                wx.CLIP_CHILDREN
        wx.Frame.__init__(self, None, title=title, style=style)

        self.Bind(wx.EVT_CLOSE, self._on_close_window)

        quit_command_id = wx.NewId()
        self.Bind(wx.EVT_MENU, self._on_quit_command, id=quit_command_id)

        save_scene_command_id = wx.NewId()
        self.Bind(wx.EVT_MENU, self._on_save_scene_command, id=save_scene_command_id)

        accelerator_table = wx.AcceleratorTable([
            (wx.ACCEL_NORMAL, wx.WXK_ESCAPE, quit_command_id),
            (wx.ACCEL_NORMAL, wx.WXK_SPACE, save_scene_command_id)
        ])
        self.SetAcceleratorTable(accelerator_table)

        self._video_panel = wx.Panel(self, size=self._image_size)
        self._video_panel.Bind(wx.EVT_ERASE_BACKGROUND,
                               self._on_video_panel_erase_background)
        self._video_panel.Bind(wx.EVT_PAINT, self._on_video_panel_paint)

        self._static_text = wx.StaticText(self)

        border = 12

        controls_sizer = wx.BoxSizer(wx.HORIZONTAL)
        controls_sizer.Add(self._static_text, 0, wx.ALIGN_CENTER_VERTICAL)

        root_sizer = wx.BoxSizer(wx.VERTICAL)
        root_sizer.Add(self._video_panel)
        root_sizer.Add(controls_sizer, 0, wx.EXPAND | wx.ALL, border)
        self.SetSizerAndFit(root_sizer)

        # Move the window to the center of the screen.
        self.Center()

        self._capture_thread = threading.Thread(target=self._run_capture_loop)
        self._running = True
        self._save_scene_pending = False
        self._capture_thread.start()


    def _on_close_window(self, event):
        self._running = False
        self._capture_thread.join()
        self.Destroy()


    def _on_quit_command(self, event):
        self.Close()


    def _on_save_scene_command(self, event):
        self._save_scene_pending = True


    def _on_video_panel_erase_background(self, event):
        pass


    def _on_video_panel_paint(self, event):

        self._rgb_image_front_buffer_lock.acquire()

        if self._rgb_image_front_buffer is None:
            self._rgb_image_front_buffer_lock.release()
            return

        # Convert the image to bitmap format.
        h, w = self._rgb_image_front_buffer.shape[:2]
        if WX_MAJOR_VERSION < 4:
            video_bitmap = wx.BitmapFromBuffer(w, h, self._rgb_image_front_buffer)
        else:
            video_bitmap = wx.Bitmap.FromBuffer(w, h, self._rgb_image_front_buffer)

        self._rgb_image_front_buffer_lock.release()

        # Show the bitmap.
        dc = wx.BufferedPaintDC(self._video_panel)
        dc.DrawBitmap(video_bitmap, 0, 0)


    def _run_capture_loop(self):
        numImagesCaptured = 0
        startTime = timeit.default_timer()
        while self._running:
            
            if self._is_monochrome:
                success, self._gray_image = self._capture.read(self._gray_image)
            else:
                success, self._bgr_image = self._capture.read(self._bgr_image)
            if success:
                numImagesCaptured += 1
                self._track_object()

                # Perform a thread-safe swap of the front and back image buffers.
                self._rgb_image_front_buffer_lock.acquire()
                self._rgb_image_front_buffer, self._rgb_image = \
                    self._rgb_image, self._rgb_image_front_buffer
                self._rgb_image_front_buffer_lock.release()

                # Signal the video panel to repaint itself from the bitmap.
                self._video_panel.Refresh()

                if self._save_scene_pending:
                    if not os.path.exists(self._saved_scenes_path):
                        os.makedirs(self._saved_scenes_path)
                    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S_%f')
                    if self._is_monochrome:
                        cv2.imwrite('%s/scene_%s_real.png' % (self._saved_scenes_path, timestamp),
                                    self._gray_image)
                    else:
                        cv2.imwrite('%s/scene_%s_real.png' % (self._saved_scenes_path, timestamp),
                                    self._bgr_image)
                    cv2.imwrite('%s/scene_%s_augmented.png' % (self._saved_scenes_path, timestamp),
                                self._rgb_image[...,::-1])
                    self._save_scene_pending = False
            deltaTime = timeit.default_timer() - startTime
            if deltaTime > 0.0:
                fps = numImagesCaptured / deltaTime
                self._init_kalman_transition_matrix(fps)
                wx.CallAfter(self._static_text.SetLabel, 'FPS:  %.1f' % fps)


    def _track_object(self):

        if self._is_monochrome:
            if self._rgb_image is None:
                h, w = self._gray_image.shape
                self._rgb_image = numpy.empty((h, w, 3), self._gray_image.dtype)
        else:
            if self._rgb_image is None:
                self._rgb_image = numpy.empty_like(self._bgr_image)
            self._gray_image = convert_to_gray(self._bgr_image, self._gray_image)

        if self._mask is None:
            self._mask = numpy.full_like(self._gray_image, 255)

        keypoints, descriptors = self._feature_detector.detectAndCompute(
            self._gray_image, self._mask)

        # Find the 2 best matches for each descriptor.
        matches = self._descriptor_matcher.knnMatch(descriptors, 2)

        # Filter the matches based on the distance ratio test.
        good_matches = [
            match[0] for match in matches
            if len(match) > 1 and match[0].distance < 0.6 * match[1].distance
        ]

        # Select the good keypoints and draw them in red.
        good_keypoints = [keypoints[match.queryIdx] for match in good_matches]
        cv2.drawKeypoints(self._gray_image, good_keypoints, self._rgb_image,
                          (255, 0, 0))

        min_good_matches_to_start_tracking = 8
        min_good_matches_to_continue_tracking = 6
        num_good_matches = len(good_matches)
        
        if num_good_matches < min_good_matches_to_continue_tracking:
            self._was_tracking = False
            self._mask.fill(255)

        elif num_good_matches >= min_good_matches_to_start_tracking or \
                self._was_tracking:

            # Select the 2D coordinates of the good matches.
            # They must be in an array of shape (N, 1, 2).
            good_points_2D = numpy.array(
                [[keypoint.pt] for keypoint in good_keypoints], FLOAT_TYPE)

            # Select the 3D coordinates of the good matches.
            # They must be in an array of shape (N, 1, 3).
            good_points_3D = numpy.array(
                [[self._reference_points_3D[match.trainIdx]] for match in good_matches],
                FLOAT_TYPE)

            # Solve for the pose and find the inlier indices.
            success, self._rotation_vector, self._translation_vector, inlier_indices = \
                cv2.solvePnPRansac(good_points_3D, good_points_2D, self._camera_matrix,
                                   self._distortion_coefficients, self._rotation_vector,
                                   self._translation_vector, useExtrinsicGuess=False,
                                   iterationsCount=100, reprojectionError=8.0,
                                   confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE)

            if success:

                if not self._was_tracking:
                    self._init_kalman_state_matrices()
                self._was_tracking = True

                self._apply_kalman()

                # Select the inlier keypoints.
                inlier_keypoints = [good_keypoints[i] for i in inlier_indices.flat]

                # Select the 2D coordinates of the inlier keypoints.
                inlier_points_2D = numpy.array(
                    [[keypoint.pt] for keypoint in inlier_keypoints], numpy.int32)

                # Draw the inlier keypoints in green.
                cv2.drawKeypoints(self._rgb_image, inlier_keypoints, self._rgb_image,
                                  (0, 255, 0))

                # Draw the axes of the tracked object.
                self._draw_object_axes()

                # Make and draw a mask around the tracked object.
                self._make_and_draw_object_mask()


    def _init_kalman_transition_matrix(self, fps):

        if fps <= 0.0:
            return

        # Velocity transition rate
        vel = 1.0 / fps

        # Acceleration transition rate
        acc = 0.5 * (vel ** 2.0)

        self._kalman.transitionMatrix = numpy.array(
            [[1.0, 0.0, 0.0, vel, 0.0, 0.0, acc, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0, vel, 0.0, 0.0, acc, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0, 0.0, vel, 0.0, 0.0, acc, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, vel, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, vel, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, vel, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, vel, 0.0, 0.0, acc, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, vel, 0.0, 0.0, acc, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, vel, 0.0, 0.0, acc],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, vel, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, vel, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, vel],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
            FLOAT_TYPE)


    def _init_kalman_state_matrices(self):

        t_x, t_y, t_z = self._translation_vector.flat
        r_x, r_y, r_z = self._rotation_vector.flat

        self._kalman.statePre = numpy.array(
            [[t_x], [t_y], [t_z], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
             [r_x], [r_y], [r_z], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]],
            FLOAT_TYPE)
        self._kalman.statePost = numpy.array(
            [[t_x], [t_y], [t_z], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
             [r_x], [r_y], [r_z], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]],
            FLOAT_TYPE)


    def _apply_kalman(self):

        self._kalman.predict()

        t_x, t_y, t_z = self._translation_vector.flat
        r_x, r_y, r_z = self._rotation_vector.flat

        estimate = self._kalman.correct(numpy.array(
            [[t_x], [t_y], [t_z], [r_x], [r_y], [r_z]], FLOAT_TYPE))

        self._translation_vector = estimate[0:3]
        self._rotation_vector = estimate[9:12]


    def _draw_object_axes(self):

        points_2D, jacobian = cv2.projectPoints(
            self._reference_axis_points_3D, self._rotation_vector,
            self._translation_vector, self._camera_matrix, self._distortion_coefficients)

        origin =  (int(points_2D[0, 0, 0]), int(points_2D[0, 0, 1]))
        right =   (int(points_2D[1, 0, 0]), int(points_2D[1, 0, 1]))
        up =      (int(points_2D[2, 0, 0]), int(points_2D[2, 0, 1]))
        forward = (int(points_2D[3, 0, 0]), int(points_2D[3, 0, 1]))

        cv2.arrowedLine(self._rgb_image, origin, right,   (255,   0,   0))  # X: red
        cv2.arrowedLine(self._rgb_image, origin, up,      (  0, 255,   0))  # Y: green
        cv2.arrowedLine(self._rgb_image, origin, forward, (  0,   0, 255))  # Z: blue


    def _make_and_draw_object_mask(self):

        # Project the object's vertices into the scene.
        vertices_2D, jacobian = cv2.projectPoints(
            self._reference_vertices_3D, self._rotation_vector, self._translation_vector,
            self._camera_matrix, self._distortion_coefficients)
        vertices_2D = vertices_2D.astype(numpy.int32)

        # Make a mask based on the projected vertices.
        self._mask.fill(0)
        for vertex_indices in self._reference_vertex_indices_by_face:
            cv2.fillConvexPoly(self._mask, vertices_2D[vertex_indices], 255)

        # Draw the mask in semi-transparent cyan.
        cv2.subtract(self._rgb_image, 16, self._rgb_image, self._mask)


def main():

    if PySpinCapture is not None:
        is_monochrome = True
        capture = PySpinCapture(0, roi=(0, 0, 960, 600), binning_radius=2,
                                is_monochrome=is_monochrome)
        diagonal_fov_degrees = 56.1  # 12.5mm lens with 1/1.2" sensor
        target_fps = 40.0
    else:
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        is_monochrome = False
        diagonal_fov_degrees = 70.0
        target_fps = 25.0

    app = wx.App()
    frame = VisualizingTheInvisible(capture, is_monochrome, diagonal_fov_degrees,
                                    target_fps)
    frame.Show()
    app.MainLoop()


if __name__ == '__main__':
    main()
