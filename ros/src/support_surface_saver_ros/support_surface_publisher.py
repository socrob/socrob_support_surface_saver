from __future__ import print_function

import numpy as np
from skimage import draw
import imageio
import os
import pickle

import rospy
import message_filters
import tf
from cv_bridge import CvBridge, CvBridgeError
import image_geometry
from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PolygonStamped, Point32, Point, PointStamped
from mbot_robot_class_ros.patterns import ConditionalSubscriber

import support_surface_object


np.warnings.filterwarnings('ignore')


class ProjectedPolygon:
    def __init__(self): self.points = []

    def __str__(self): return '\n' + '\n'.join(map(str, self.points))


class SupportSurfacePublisher:

    def __init__(self):

        # Parameters

        if not rospy.has_param("~fixed_frame"):
            rospy.logfatal("Parameter fixed_frame is missing")
            rospy.signal_shutdown("missing parameters")

        if not rospy.has_param("~support_surfaces"):
            rospy.logfatal("Parameter support_surfaces is missing")
            rospy.signal_shutdown("missing parameters")

        if not isinstance(rospy.get_param("~support_surfaces"), dict):
            rospy.logfatal("Parameter support_surfaces should be a dictionary {polygon_id: [[x, y, z], ...], ...}")
            rospy.signal_shutdown("missing parameters")

        self.publish_polygons = rospy.get_param("~publish_polygons", True)

        self.max_depth = rospy.get_param("~max_depth", 4.0)
        self.max_rate = rospy.get_param("~max_rate", 10.0)

        self.fixed_frame = rospy.get_param("~fixed_frame")
        self.polygon_ids = rospy.get_param("~support_surfaces").keys()
        self.polygons = list()
        for surface_id in self.polygon_ids:
            poly_fixed = PolygonStamped()
            poly_fixed.header.frame_id = self.fixed_frame
            poly_fixed.polygon.points = map(lambda l: Point32(*l), rospy.get_param("~support_surfaces")[surface_id])
            self.polygons.append(poly_fixed)

        # Variables
        self.bridge = CvBridge()
        self.listener = tf.TransformListener()
        self.enable_process = False

        self.conditional_subscriber = ConditionalSubscriber()

        # Subscribers
        self.camera_info_sub = self.conditional_subscriber.make_subscriber("~camera_info_in", CameraInfo, self.camera_callback, queue_size=1)

        # Publishers
        self.depth_pub = self.conditional_subscriber.make_publisher("support_surface/depth/image", Image, queue_size=1)
        self.depth_info_pub = self.conditional_subscriber.make_publisher("support_surface/depth/camera_info", CameraInfo, queue_size=1)

        # polygon_pubs has polygon_ids as keys, and corresponding publishers as values
        self.polygon_pubs = dict(zip(self.polygon_ids,
                                     map(lambda polygon_id: self.conditional_subscriber.make_publisher("support_surface_poly/%s" % polygon_id, PolygonStamped, queue_size=1, latch=True),
                                         self.polygon_ids)))

    def loop(self):
        r = rospy.Rate(self.max_rate)
        while not rospy.is_shutdown():
            r.sleep()
            self.enable_process = True

    def camera_callback(self, camera_info):
        if self.enable_process:
            self.enable_process = False
            self.process_camera_info(camera_info)

    def process_camera_info(self, camera_info):

        camera = image_geometry.PinholeCameraModel()
        camera.fromCameraInfo(camera_info)

        image_shape = (camera_info.height, camera_info.width)
        depth = np.empty(image_shape, dtype=np.float32)
        depth[:] = np.inf

        for poly_fixed, polygon_id in zip(self.polygons, self.polygon_ids):

#            self.polygon_pubs[polygon_id].publish(poly_fixed)

            poly_camera = PolygonStamped()
            poly_camera.header.frame_id = camera_info.header.frame_id
            poly_camera.polygon.points = []
            poly_image = ProjectedPolygon()

            try:
                self.listener.waitForTransform(poly_fixed.header.frame_id, camera_info.header.frame_id,
                                               camera_info.header.stamp, rospy.Duration(1.0/self.max_rate))

            except tf.Exception:
                return

            for p in poly_fixed.polygon.points:

                # transform the polygon in camera frame
                point_fixed = PointStamped(header=poly_fixed.header, point=p)
                point_camera = self.listener.transformPoint(camera_info.header.frame_id, point_fixed)
                poly_camera.polygon.points.append(point_camera.point)

                # transform the polygon in image coordinates
                u, v = camera.project3dToPixel((point_camera.point.x, point_camera.point.y, point_camera.point.z))
                poly_image.points.append(np.array([u, v, point_camera.point.z]))

            image_points = np.array(poly_image.points)
            depth_bounded_points = image_points[image_points[:, 2] > 0, :]
            depth_bounded_points = depth_bounded_points[depth_bounded_points[:, 2] < self.max_depth, :]
            camera_points = np.array(map(lambda p: np.array([p.x, p.y, p.z]), poly_camera.polygon.points))

            if len(depth_bounded_points) >= 3:

                # compute polygon mask for positive-depth points of the polygon
                poly_mask = np.zeros(image_shape, dtype=np.bool)
                poly_mask_rows, poly_mask_cols = draw.polygon(depth_bounded_points[:, 1],
                                                              depth_bounded_points[:, 0], image_shape)
                poly_mask[poly_mask_rows, poly_mask_cols] = True

                # compute surface plane from 3 points of the polygon in camera frame
                p1, p2, p3 = camera_points[0:3]

                # these two vectors are in the plane
                v1, v2 = p3 - p1, p2 - p1

                # the cross product is a vector normal to the plane
                n_x, n_y, n_z = surface_normal = np.cross(v1, v2)

                # this evaluates a*p3.x + b*p3.y + c*p3.z, which equals d
                d = np.dot(surface_normal, p3)

                # get (u, v) coordinates for each pixel in the image, as array
                u, v = np.array(np.meshgrid(range(camera_info.width), range(camera_info.height)))

                # compute the vector (x, y, z) for the ray passing through each pixel (like camera.projectPixelTo3dRay(uv))
                x = (u - camera.cx()) / camera.fx()
                y = (v - camera.cy()) / camera.fy()
                z = np.ones(image_shape)

                # compute intersection between surface plane and each ray, on z axis only
                surface_z = z * d / (x*n_x + y*n_y + z*n_z)

                poly_depth = np.empty(image_shape, dtype=np.float32)
                poly_depth[:] = np.nan
                poly_depth[poly_mask] = surface_z[poly_mask]
                poly_depth[surface_z <= 0] = np.nan

                # z_mask is True where:
                # poly is not nan (only the pixels of poly are copied, not the nan values) and
                # poly_depth is closer that depth (z-buffer) and
                # depth is inf (poly_depth is copied when depth is not initialised)
                z_mask = poly_depth < depth
                depth[z_mask] = poly_depth[z_mask]

        # encode images to save and publish
        depth[depth == np.inf] = np.nan
        try:
            support_surface_msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
            support_surface_msg.header = camera_info.header
        except CvBridgeError as e:
            rospy.logerr(e)
            return


        self.depth_pub.publish(support_surface_msg)
        self.depth_info_pub.publish(camera_info)

