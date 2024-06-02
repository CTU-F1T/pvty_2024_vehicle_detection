import rospy
import math
import numpy as np
from sensor_msgs.msg import Image, LaserScan, CameraInfo
import cv2
from cv_bridge import CvBridge
import message_filters
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped
import tf
import apriltag
import time

class Scan:
    """Class to handle LaserScan data"""
    def __init__(self):
        self.scan = None
        # Subscribe to the LaserScan topic
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)

    def scan_callback(self, scan):
        """Callback to receive scan data"""
        self.scan = scan

    def process_scan(self, scan):
        """Process scan data to extract coordinates of valid points"""
        start_angle, end_angle = scan.angle_min, scan.angle_max
        a_inc = scan.angle_increment
        ranges, max_rng, min_rng = scan.ranges, scan.range_max, scan.range_min

        coords = []
        c_angle = start_angle
        for r in ranges:
            if self.is_valid(max_rng, min_rng, r):
                # Convert polar coordinates to Cartesian coordinates
                x, y = r * np.cos(c_angle), r * np.sin(c_angle)
                coords.append((x, y))
            c_angle += a_inc

        return coords

    def is_valid(self, max_r, min_r, r):
        """Check if the range is within valid bounds"""
        return min_r < r < max_r


class AprilTagDetection:
    """Class to handle AprilTag detection and pose estimation"""
    def __init__(self):
        options = apriltag.DetectorOptions(families="tag36h11")
        self.at_detector = apriltag.Detector(options)
        self.tag_size = 0.1
        self.camera_matrix, self.dist_coeffs = None, None
        self.last_depth = None

    def camera_info_callback(self, msg):
        """Callback to receive camera info"""
        self.camera_matrix = np.array(msg.K).reshape(3, 3)
        self.dist_coeffs = np.array(msg.D)

    def detect_and_estimate_pose(self, gray_img):
        """Detect AprilTags and estimate their pose"""
        return self.at_detector.detect(gray_img)

    def rotation_matrix_to_euler_angles(self, R):
        """Convert rotation matrix to Euler angles"""
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            x, y, z = np.arctan2(R[2, 1], R[2, 2]), np.arctan2(-R[2, 0], sy), np.arctan2(R[1, 0], R[0, 0])
        else:
            x, y, z = np.arctan2(-R[1, 2], R[1, 1]), np.arctan2(-R[2, 0], sy), 0

        return np.array([x, y, z])

    def euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles to quaternion"""
        return tf.transformations.quaternion_from_euler(0, 0, -pitch)


class ImageProcessing:
    """Main class to handle image and scan data processing"""
    def __init__(self, visualise=False):
        self.bridge = CvBridge()
        self.visualise = visualise
        self.scan = Scan()
        self.april_tag_detection = AprilTagDetection()
        self.P = np.array([[604.3504028320312, 0.0, 321.988525390625, 0.0],
                           [0.0, 604.9443359375, 241.68743896484375, 0.0],
                           [0.0, 0.0, 1.0, 0.0]])

        self.setup_subscribers()
        self.setup_publishers()
        rospy.loginfo('Node initialized')

    def setup_subscribers(self):
        """Setup all necessary subscribers"""
        self.im_sub = message_filters.Subscriber('/camera/color/image_raw', Image, buff_size=2**24)
        self.depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, buff_size=2**24)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.im_sub, self.depth_sub], queue_size=1, slop=0.5)
        self.ts.registerCallback(self.process_data)
        rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.camera_info_callback)

    def setup_publishers(self):
        """Setup all necessary publishers"""
        self.marker_pub = rospy.Publisher('/detection/marker', Marker, queue_size=1)
        self.grad_pub = rospy.Publisher('/gradient/merged_grad', Image, queue_size=1)
        self.grad_im_pub = rospy.Publisher('/gradient/image_grad', Image, queue_size=1)
        self.grad_d_pub = rospy.Publisher('/gradient/depth_grad', Image, queue_size=1)
        self.focus_pub = rospy.Publisher('/detection/focus', Image, queue_size=1)
        self.at_pub = rospy.Publisher('/detection/april_tags', Image, queue_size=1)
        self.at_marker_pub = rospy.Publisher('/detection/at_marker', Marker, queue_size=1)
        self.pose_pub = rospy.Publisher('/detection/pose_estimate', PoseStamped, queue_size=1)

    def camera_info_callback(self, msg):
        """Callback to receive camera info and forward it to AprilTagDetection"""
        self.april_tag_detection.camera_info_callback(msg)

    def process_data(self, img, depth):
        """Main callback to process synchronized image and depth data"""
        start_time = time.time()

        # Convert ROS image messages to OpenCV images
        img_data = self.bridge.imgmsg_to_cv2(img, desired_encoding="passthrough")
        depth_data = self.bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough")
        
        # Process scan data
        scan = self.scan.scan
        if scan is None:
            rospy.logwarn('Scan data not available')
            return -1
        scan = self.scan.process_scan(scan)

        # Convert the RGB image to grayscale and blur the images
        gray_img = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
        blurred_depth = cv2.GaussianBlur(depth_data, (13, 13), 1)
        grad_blurred = self.get_grad(blurred_depth)
        gray_img = cv2.GaussianBlur(gray_img, (13, 13), 1)
        grad_im = self.get_grad(gray_img)

        # Merge gradients and publish results
        merged_grad = self.merge_gradients(grad_blurred, grad_im, depth_data)
        self.publish_gradients(merged_grad, grad_im, grad_blurred)

        # Detect AprilTags and publish detections
        tags = self.april_tag_detection.detect_and_estimate_pose(gray_img)
        self.publish_april_tag_detections(tags, img_data)

        rospy.loginfo(f'Processing time: {time.time() - start_time}s')

    def get_grad(self, data):
        """Calculate gradient magnitude of the given data"""
        kernel_x, kernel_y = np.array([[-1, 1]]), np.array([[-1], [1]])
        grad_x, grad_y = cv2.filter2D(data, -1, kernel_x), cv2.filter2D(data, -1, kernel_y)
        return np.sqrt(grad_x ** 2 + grad_y ** 2)

    def merge_gradients(self, grad_blurred, grad_im, depth_data):
        """Merge image and depth gradients"""
        th_gi = 8
        th_idx_gim = grad_im > th_gi
        grad_im_filt = np.zeros_like(grad_im)
        grad_im_filt[th_idx_gim] = 1

        th = 10
        th_idx_b = grad_blurred > th
        grad_filt_b = np.zeros_like(grad_blurred)
        grad_filt_b[th_idx_b] = 1

        # Create an interest mask to focus on central region
        interest_mask = np.ones_like(depth_data)
        interest_mask[:50, :] = 0
        interest_mask[380:, :] = 0

        return cv2.bitwise_and(grad_filt_b, grad_im_filt.astype(np.float32), mask=interest_mask.astype(np.uint8))

    def publish_gradients(self, merged_grad, grad_im, grad_blurred):
        """Publish gradient images to corresponding topics"""
        self.grad_pub.publish(self.bridge.cv2_to_imgmsg(merged_grad, encoding="passthrough"))
        self.grad_im_pub.publish(self.bridge.cv2_to_imgmsg(255 * grad_im.astype(np.uint8), encoding="passthrough"))
        self.grad_d_pub.publish(self.bridge.cv2_to_imgmsg(255 * grad_blurred.astype(np.uint8), encoding="passthrough"))

    def publish_april_tag_detections(self, tags, img_data):
        """Publish AprilTag detections and their poses"""
        at_image = np.copy(img_data)
        for tag in tags:
            # Draw tag boundaries and ID on the image
            corners, tag_center = tag.corners, tag.center
            for i in range(4):
                pt1, pt2 = (int(corners[i][0]), int(corners[i][1])), (int(corners[(i + 1) % 4][0]), int(corners[(i + 1) % 4][1]))
                cv2.line(at_image, pt1, pt2, (0, 255, 0), 2)
            center = (int(tag_center[0]), int(tag_center[1]))
            cv2.putText(at_image, str(tag.tag_id), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Estimate pose of the detected tag
            pose, e0, e1 = self.april_tag_detection.at_detector.detection_pose(
                tag, 
                camera_params=(self.april_tag_detection.camera_matrix[0, 0], 
                               self.april_tag_detection.camera_matrix[1, 1], 
                               self.april_tag_detection.camera_matrix[0, 2], 
                               self.april_tag_detection.camera_matrix[1, 2]), 
                tag_size=self.april_tag_detection.tag_size
            )

            # Convert rotation matrix to Euler angles
            euler_angles = self.april_tag_detection.rotation_matrix_to_euler_angles(pose[:3, :3])
            e_angl_deg = np.degrees(euler_angles)
            rospy.loginfo(f'ID {tag.tag_id} | angles: {e_angl_deg}')

        # Publish the image with detected tags
        self.at_pub.publish(self.bridge.cv2_to_imgmsg(at_image.astype(np.uint8)))

    def interpolate(self, depth_data):
        """Interpolate missing depth data (zero values)"""
        zero_indices = (depth_data == 0)
        mask = ~zero_indices
        x, y = np.meshgrid(np.arange(depth_data.shape[1]), np.arange(depth_data.shape[0]))

        interpolated_values = np.zeros_like(depth_data)
        for i in range(depth_data.shape[0]):
            if np.any(mask[i]):
                interpolated_values[i] = np.interp(x[i], x[i][mask[i]], depth_data[i][mask[i]])
            else:
                return depth_data
        return interpolated_values

    def get_centroid_it(self, points):
        """Iteratively find the centroid of a set of points"""
        x, y = points[0], points[1]
        num_it, w, h = 1, 200, 100

        if len(x) == 0:
            rospy.logwarn('No points')
            return None, None

        for _ in range(num_it):
            centroid_x, centroid_y = int(np.mean(x)), int(np.mean(y))
            left_boundary, right_boundary = centroid_x - w / 2, centroid_x + w / 2
            top_boundary, bottom_boundary = centroid_y - h / 2, centroid_y + h / 2

            # Filter points inside the rectangle
            inside_rectangle_indexes = np.where((x >= left_boundary) & (x <= right_boundary) &
                                                (y >= top_boundary) & (y <= bottom_boundary))[0]
            x, y = x[inside_rectangle_indexes], y[inside_rectangle_indexes]

        return centroid_x, centroid_y

    def pix_to_camera(self, pixel, depth=1):
        """Convert pixel coordinates to camera coordinates"""
        u, v = pixel[1], pixel[0]
        fx, fy, cx, cy = self.P[0, 0], self.P[1, 1], self.P[0, 2], self.P[1, 2]

        x = depth / 1000
        y = -(u - cx) * (depth / 1000) / fx
        point = [x, y]
        return point

    def bfs_filter(self, depth, image, d0, c):
        """Apply a BFS filter to the image based on depth and color similarity"""
        if d0 == 1:
            return image

        mask, visited = np.zeros_like(depth), np.zeros_like(depth)
        chunk, surr = 10, [(-chunk, 0), (chunk, 0), (0, chunk), (0, -chunk), (chunk, chunk), (-chunk, -chunk), (-chunk, chunk), (chunk, -chunk)]
        q, dist_d, dist_c = [c], 250, 100

        mask[c[0] - chunk:c[0] + chunk, c[1] - chunk:c[1] + chunk] = 1
        visited[c[0], c[1]] = 1

        while q:
            u = np.array(q.pop(0))
            q_depth, q_col = depth[c[0], c[1]], image[c[0], c[1]]

            for s in surr:
                v = u + s
                if 0 <= v[0] < depth.shape[0] and 0 <= v[1] < depth.shape[1]:
                    tmp_d, tmp_col = depth[v[0], v[1]], image[c[0], c[1]]
                    if abs(tmp_d - q_depth) <= dist_d and self.rgb_close(q_col, tmp_col) <= dist_c:
                        if visited[v[0], v[1]] == 0:
                            q.append(v)
                            mask[v[0] - chunk:v[0] + chunk, v[1] - chunk:v[1] + chunk] = 1
                            visited[v[0], v[1]] = 1

        mask = np.expand_dims(mask, axis=2)
        return image * mask

    def rgb_close(self, p1, p2):
        """Check if two RGB colors are close to each other"""
        return np.sqrt(np.sum((p1 - p2) ** 2))


if __name__ == '__main__':
    rospy.init_node('Image_node', anonymous=True)
    ImageProcessing(visualise=True)
    rospy.spin()
