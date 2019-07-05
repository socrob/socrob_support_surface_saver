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

import support_surface_object


np.warnings.filterwarnings('ignore')


class ProjectedPolygon:
    def __init__(self): self.points = []

    def __str__(self): return '\n' + '\n'.join(map(str, self.points))


class SupportSurfaceSaver:

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

        self.save_images = rospy.get_param("~save_images", True)
        self.publish_images = rospy.get_param("~publish_images", True)
        self.publish_depth = rospy.get_param("~publish_depth", True)
        self.publish_polygons = rospy.get_param("~publish_polygons", True)

        image_folder_output = os.path.expanduser(rospy.get_param("~output_folder", "~/support_surface_output"))
        self.save_images_every = rospy.get_param("~save_images_every", 30)
        self.max_depth = rospy.get_param("~max_depth", 4.)

        self.fixed_frame = rospy.get_param("~fixed_frame")
        self.polygon_ids = rospy.get_param("~support_surfaces").keys()
        self.polygons = list()
        for surface_id in self.polygon_ids:
            poly_fixed = PolygonStamped()
            poly_fixed.header.frame_id = self.fixed_frame
            poly_fixed.polygon.points = map(lambda l: Point32(*l), rospy.get_param("~support_surfaces")[surface_id])
            self.polygons.append(poly_fixed)

        # Variables
        self.i = 0
        self.images_counter = 0
        self.bridge = CvBridge()
        self.listener = tf.TransformListener()
        self.image_file_format_string = os.path.join(image_folder_output, '%05i_image.png')
        self.depth_file_format_string = os.path.join(image_folder_output, '%05i_depth.png')
        self.pickle_file_format_string = os.path.join(image_folder_output, '%05i_pickle.pkl')

        # Subscribers
        self.camera_info_sub = message_filters.Subscriber("~camera_info_in", CameraInfo)
        self.image_sub = message_filters.Subscriber("~images_in", Image)
        self.time_synchronizer = message_filters.TimeSynchronizer([self.image_sub, self.camera_info_sub], queue_size=1)
        self.time_synchronizer.registerCallback(self.camera_callback)

        # Publishers
        if self.publish_depth:
            self.depth_pub = rospy.Publisher("support_surface/depth/image", Image, queue_size=1)
            self.depth_info_pub = rospy.Publisher("support_surface/depth/camera_info", CameraInfo, queue_size=1)
        else:
            self.depth_pub = None
            self.depth_info_pub = None

        if self.publish_images:
            self.image_pub = rospy.Publisher("support_surface/rgb/image_rect_color", Image, queue_size=1)
            self.image_info_pub = rospy.Publisher("support_surface/rgb/camera_info", CameraInfo, queue_size=1)
        else:
            self.image_pub = None
            self.image_info_pub = None

        
        # polygon_pubs has polygon_ids as keys, and corresponding publishers as values
        self.polygon_pubs = dict(zip(self.polygon_ids,
                                     map(lambda polygon_id: rospy.Publisher("support_surface_poly/%s" % polygon_id,
                                                                            PolygonStamped, queue_size=1, latch=True),
                                         self.polygon_ids)))
        
        # Make sure the output folder exists
        if not os.path.exists(image_folder_output):
            rospy.logwarn("mkdir %s" % image_folder_output)
            os.mkdir(image_folder_output)

    def image_filename(self, index):
        return self.image_file_format_string % index

    def depth_filename(self, index):
        return self.depth_file_format_string % index

    def pickle_filename(self, index):
        return self.pickle_file_format_string % index

    @staticmethod
    def loop():
        rospy.spin()

    def camera_callback(self, image, camera_info):
        self.process_camera_info(image, camera_info)

    def process_camera_info(self, image_msg, camera_info):

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
                                               camera_info.header.stamp, rospy.Duration(0.1))

                origin_fixed = PointStamped(header=Header(frame_id=self.fixed_frame), point=Point(0, 0, 0))
                origin_camera = self.listener.transformPoint(camera_info.header.frame_id, origin_fixed)
                z_vector_fixed = PointStamped(header=Header(frame_id=self.fixed_frame), point=Point(0, 0, 1))
                z_vector_camera = self.listener.transformPoint(camera_info.header.frame_id, z_vector_fixed)
                surface_normal_camera = (z_vector_camera.point.x - origin_camera.point.x,
                                         z_vector_camera.point.y - origin_camera.point.y,
                                         z_vector_camera.point.z - origin_camera.point.z)

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
            positive_points = image_points[image_points[:, 2] > 0, :]
            camera_points = np.array(map(lambda p: np.array([p.x, p.y, p.z]), poly_camera.polygon.points))

            if len(positive_points) >= 3:

                # compute polygon mask for positive-depth points of the polygon
                poly_mask = np.zeros(image_shape, dtype=np.bool)
                poly_mask_rows, poly_mask_cols = draw.polygon(positive_points[:, 1],
                                                              positive_points[:, 0], image_shape)
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
            image = self.bridge.imgmsg_to_cv2(image_msg, "rgb8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        if self.publish_depth:
            self.depth_pub.publish(support_surface_msg)
            self.depth_info_pub.publish(camera_info)

        if self.publish_images:
            self.image_pub.publish(image_msg)
            self.image_info_pub.publish(camera_info)

        # save image and depth to files if there is a support surface in the image
        if self.save_images and \
                self.images_counter % self.save_images_every == 0 and \
                np.sum(np.logical_not(np.isnan(depth))) > 0:

            while os.path.exists(self.image_filename(self.i)) or os.path.exists(self.depth_filename(self.i)):
                self.i += 1

            imageio.imwrite(self.image_filename(self.i), image)
            rospy.loginfo("saved image to %s" % self.image_filename(self.i))

            depth_image = np.zeros(depth.shape, dtype=depth.dtype)
            depth_mask = depth < self.max_depth
            depth_image[depth_mask] = (depth * 255./self.max_depth)[depth_mask]
            imageio.imwrite(self.depth_filename(self.i), depth_image)
            rospy.loginfo("saved depth to %s" % self.depth_filename(self.i))

            with open(self.pickle_filename(self.i), 'wb') as f:
                pickle.dump(support_surface_object.SupportSurface(image, depth, camera_info, surface_normal_camera), f, pickle.HIGHEST_PROTOCOL)
            rospy.loginfo("saved pickle to %s" % self.pickle_filename(self.i))

            self.i += 1

        self.images_counter += 1

