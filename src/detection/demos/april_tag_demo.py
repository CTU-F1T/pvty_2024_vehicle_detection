"""
    Simple subscriber to show and test how to work with camera topics
"""

import rospy
import math
import numpy as np
from sensor_msgs.msg import Image
import matplotlib.pyplot as plt
import cv2
from cv_bridge import CvBridge
import message_filters      # for calback data synchronisation
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
import matplotlib.pyplot as plt
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseStamped
import tf

import apriltag
import time


class ProcessData:  # rename to detect?
    """
    Class to process the data
    """

    def __init__(self,visualise=False) -> None:
        rospy.loginfo('Initializing node')
        self.initialized = False

        self.im_sub = message_filters.Subscriber('/camera/color/image_raw',Image,buff_size=2**24)
        self.depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw',Image,buff_size=2**24)
        #self.scan_sub = message_filters.Subscriber('/scan_dummy',LaserScan)
        self.scan_sub = rospy.Subscriber('/scan',LaserScan,self.scancb)
        #message_filters.Subscriber('/scan',LaserScan)
        #self.ts = message_filters.ApproximateTimeSynchronizer([self.im_sub, self.depth_sub, self.scan_sub], queue_size=10, slop=1)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.im_sub,self.depth_sub],queue_size = 1, slop=0.5)
        # self.ts = message_filters.ApproximateTimeSynchronizer([self.scan_sub], queue_size=10, slop=0.5)
        self.ts.registerCallback(self.process_data)

        rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.camera_info_callback)

        self.marker_pub = rospy.Publisher('/detection/marker',Marker, queue_size=1)
        self.grad_pub = rospy.Publisher('/gradient/merged_grad',Image,queue_size=1)
        self.grad_im_pub = rospy.Publisher('/gradient/image_grad',Image,queue_size=1)
        self.grad_d_pub = rospy.Publisher('/gradient/depth_grad',Image,queue_size=1)
        self.focus_pub = rospy.Publisher('/detection/focus',Image,queue_size=1)
        self.at_pub = rospy.Publisher('/detection/april_tags',Image,queue_size=1)
        self.at_marker_pub = rospy.Publisher('/detection/at_marker',Marker, queue_size=1)
        self.pose_pub = rospy.Publisher('/detection/pose_estimate', PoseStamped, queue_size=1)


        self.vis = visualise
        self.bridge = CvBridge()
        self.scan = None

        self.last_depth = None
        
        self.detect_r = 100
        self.init_rw = 300
        self.init_rh = 150
        
        self.rec_w = 200
        self.rec_h = 100

        self.P = np.array([[604.3504028320312, 0.0, 321.988525390625,0.0],[0.0, 604.9443359375, 241.68743896484375, 0.0], [0.0, 0.0, 1.0, 0.0]])


        options = apriltag.DetectorOptions(families="tag36h11")

        self.at_detector = apriltag.Detector(options)
        self.tag_size = 0.1


        self.initialized = True
        rospy.loginfo('Node initialized')

    def camera_info_callback(self,msg):
        self.camera_matrix = np.array(msg.K).reshape(3, 3)
        # Extract the distortion coefficients
        self.dist_coeffs = np.array(msg.D)
        # print('CAMinfo',self.camera_matrix,self.dist_coeffs)

    def process_data(self,img:Image,depth:Image):
    # def process_data(self,scan:LaserScan):
        """
        input: image: image <Image> from the camera
               depth: depth <Image> from the camera lidar, aligned 
               scan: <LaserScan> from the planar lidar
        """
        # rospy.loginfo('process data cb')        
        # POINT_OF_INTEREST = [0.9, 0.1]
        start_time = time.time()


        FILTER_RADIUS = 0.6

        img_data = self.bridge.imgmsg_to_cv2(img,desired_encoding="passthrough")     # converts to image (rgb8), to visualise with cv2 in true color use bgr8
        depth_data = self.bridge.imgmsg_to_cv2(depth,desired_encoding="passthrough") # 16UC1
        scan = self.scan
        if scan is None:
          print('scan is none')
          return -1
        scan = self.process_scan(scan)


        # IMAGE PROCESSING #######################################################################################################

        gray_img = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)                       # grayscale image

        blurred_depth = cv2.GaussianBlur(depth_data, (13, 13), 1)                   # depth filtered with gaussian 
        grad_blurred = self.get_grad(blurred_depth)                                 # magnitude of x,y gradients

        gray_img = cv2.GaussianBlur(gray_img, (13, 13), 1)                          # grayscale filtered with gaussian 
        grad_im = self.get_grad(gray_img)                                           # magnitude of x,y gradients
    
        th_gi = 8                                                                   # grayscale threshold
        th_idx_gim = grad_im>th_gi
        grad_im_filt= np.zeros_like(grad_im)
        grad_im_filt[th_idx_gim] = 1

        th = 10                                                                     # gradient threshold
        th_idx_b = grad_blurred>th
        grad_filt_b= np.zeros_like(grad_blurred)                                    # thresholds blurred gradient
        grad_filt_b[th_idx_b] = 1

        interest_mask = np.ones_like(depth_data)                                    # filters out the region of interest
        interest_mask[:50,:] = 0                
        interest_mask[380:,:] = 0

        merged_grad = cv2.bitwise_and(grad_filt_b,grad_im_filt.astype(np.float32),mask=interest_mask.astype(np.uint8)) # merges the gradients together

        # mg_im = Image()
        # mg_im.header.stamp = rospy.Time.now()
        mg_im= self.bridge.cv2_to_imgmsg(merged_grad, encoding="passthrough")
        # mg_im.width = merged_grad.shape[1]
        # mg_im.height = merged_grad.shape[0]
        self.grad_pub.publish(mg_im)

        ig_im= self.bridge.cv2_to_imgmsg(255*grad_im_filt.astype(np.int8), encoding="passthrough")
        self.grad_im_pub.publish(ig_im)

        dg_im= self.bridge.cv2_to_imgmsg(255*grad_filt_b.astype(np.int8), encoding="passthrough")
        self.grad_d_pub.publish(dg_im)

        # cv2.imshow('Image gradient',grad_im_filt.astype(np.float32))
        # cv2.imshow('Depth gradient',grad_filt_b)
        # cv2.waitKey(0)

        tags = self.at_detector.detect(gray_img)

        # print(tags)
        # Draw the detected tags and estimate their poses
        at_image = np.copy(img_data)
        for tag in tags:
            corners = tag.corners
            # print('CT',tag.center)
            tag_center = tag.center
            for i in range(4):
                pt1 = (int(corners[i][0]), int(corners[i][1]))
                pt2 = (int(corners[(i + 1) % 4][0]), int(corners[(i + 1) % 4][1]))
                cv2.line(at_image, pt1, pt2, (0, 255, 0), 2)
            center = (int(tag.center[0]), int(tag.center[1]))
            cv2.putText(at_image, str(tag.tag_id), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Estimate the pose of the tag
            pose, e0, e1 = self.at_detector.detection_pose(
                tag, 
                camera_params=(
                    self.camera_matrix[0, 0], 
                    self.camera_matrix[1, 1], 
                    self.camera_matrix[0, 2], 
                    self.camera_matrix[1, 2]
                ), 
                tag_size=self.tag_size
            )

            rvec = pose[:3, :3]
            tvec = pose[:3, 3]

            euler_angles = self.__rotation_matrix_to_euler_angles(pose[:3, :3])
            e_angl_deg = np.degrees(euler_angles)

            tag_dist = tvec[2]

            # Print the pose information
            # print(f"Tag ID: {tag.tag_id}")
            # print("Rotation Matrix (rvec):\n", rvec)
            # print("Translation Vector (tvec):\n", tvec)
            # print("Euler Angles (radians):\n", euler_angles)
            # print("Euler Angles (degrees):\n", np.degrees(euler_angles))
            print(f'ID {tag.tag_id}|angles: {e_angl_deg}')

        at_im_pub = self.bridge.cv2_to_imgmsg(at_image.astype(np.uint8))
        self.at_pub.publish(at_im_pub)

        # # Display the image with detected tags
        # cv2.imshow('AprilTag Detection', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
   

        if len(tags):
            # print('TC',tag_center)
            centroid_row,centroid_col = int(tag_center[1]),int(tag_center[0])   
            # PUBLISH MARKER - lidar
            AT_POINT_OF_INTEREST = [centroid_row,centroid_col]
            # potreba pretransofrmovat do lidar frameP
            # depth_int = depth_data[centroid_row,centroid_col] if depth_data[centroid_row,centroid_col] != 0 else tag_dist
            # depth_int = int((tag_dist*1000))
            depth_int = int((tag_dist*1000)*np.cos(40*(np.pi/180)))
            if depth_int == 0 and self.last_depth is not None:
                depth_int = self.last_depth
            self.last_depth = depth_int
            AT_POINT_OF_INTEREST = self.pix_to_camera(AT_POINT_OF_INTEREST,depth=depth_int)

            # PUBLISH APRIL TAG MARKER ###############################################################################################
            at_marker = Marker()
            at_marker.header.frame_id = "laser"
            at_marker.header.stamp = rospy.Time.now()
            at_marker.type = 2
            at_marker.id = 1
            # Set the scale of the marker
            at_marker.scale.x = 0.1
            at_marker.scale.y = 0.1
            at_marker.scale.z = 0.1
            # Set the color
            at_marker.color.r = 0.0
            at_marker.color.g = 0.0
            at_marker.color.b = 1.0
            at_marker.color.a = 1.0
            # Set the pose of the marker
            at_marker.pose.position.x = AT_POINT_OF_INTEREST[0]
            at_marker.pose.position.y = AT_POINT_OF_INTEREST[1]
            at_marker.pose.position.z = 0
            at_marker.pose.orientation.x = 0.0
            at_marker.pose.orientation.y = 0.0
            at_marker.pose.orientation.z = 0.0
            at_marker.pose.orientation.w = 0.0

            self.at_marker_pub.publish(at_marker)     

            # PUBLISH ORIENTATION #######################################################################################################
            pose_msg = PoseStamped()
            # Fill in the header
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = 'laser'  # Change to the appropriate frame
            # Set the position
            pose_msg.pose.position.x = AT_POINT_OF_INTEREST[0]
            pose_msg.pose.position.y = AT_POINT_OF_INTEREST[1]
            pose_msg.pose.position.z = 0
            # Convert the Euler angles to a quaternion
            quaternion = self.__euler_to_quaternion(
                euler_angles[0],
                euler_angles[1],
                euler_angles[2]
            )
            # Set the orientation
            pose_msg.pose.orientation.x = quaternion[0]
            pose_msg.pose.orientation.y = quaternion[1]
            pose_msg.pose.orientation.z = quaternion[2]
            pose_msg.pose.orientation.w = quaternion[3]

            # Publish the pose
            self.pose_pub.publish(pose_msg)


        else:
            centroid_row,centroid_col = self.get_centroid_it(np.where(merged_grad==1))  # gets the centroid of the points   


        if centroid_row != None:
            cr = 10        # set the  width for the mean
            d_ker = depth_data[centroid_row-cr:centroid_row+cr,centroid_col-cr:centroid_col+cr]
            ker_min = np.min(d_ker)
            ker_th = 500
            ker_idx = np.where(d_ker-ker_min<=ker_th)
            depth_mean = np.mean(d_ker[ker_idx])   # get the mean of depth in the centroid

            d_margin_l = 100
            d_margin_h = 400
            depth_mask = np.where((depth_data<depth_mean+d_margin_h)&(depth_data>depth_mean-d_margin_l))    # filter out pixels within some dist range

            image_final = np.zeros_like(img_data)
            image_final[depth_mask] = img_data[depth_mask]

            rec_w = self.rec_w # rec width
            rec_h = self.rec_h # rec height
            # FIXME: adaptive rectangle parameters, set here? Based on the last centroind distance, initialize to some value first?
            # self.rec_w = self.init_rw*(depth_mean/1000)
            # self.rec_h = self.init_rh*(depth_mean/1000)

            POINT_OF_INTEREST = [centroid_row,centroid_col]
            # potreba pretransofrmovat do lidar frame
            depth_int = depth_data[centroid_row,centroid_col] if depth_data[centroid_row,centroid_col] != 0 else 1
            if depth_int == 0 and self.last_depth is not None:
                depth_int = self.last_depth
            self.last_depth = depth_int
            POINT_OF_INTEREST = self.pix_to_camera(POINT_OF_INTEREST,depth=depth_int)


            # focus_im = self.bfs_filter(depth_data,img_data,depth_int,[centroid_row,centroid_col])
            # focus_im_pub =  self.bridge.cv2_to_imgmsg(focus_im.astype(np.uint8))
            # self.focus_pub.publish(focus_im_pub)

            # plt.imshow(focus_im)
            # plt.show()
            # if focus_im is not None:
            #     cv2.imshow(focus_im)
            #     cv2.waitKey(0)
            # print('fs',focus_im.shape,type(focus_im))
            # cv2_foc = focus_im.astype(np.uint8)
            # print('cv2',type(cv2_foc))
            # cv2.imshow(cv2_foc)
            # cv2.waitKey(0)

            # PUBLISH MARKER - lidar

            marker = Marker()

            marker.header.frame_id = "laser"
            marker.header.stamp = rospy.Time.now()

            # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
            marker.type = 2
            marker.id = 0

            # Set the scale of the marker
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1

            # Set the color
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            # Set the pose of the marker
            marker.pose.position.x = POINT_OF_INTEREST[0]
            marker.pose.position.y = POINT_OF_INTEREST[1]
            marker.pose.position.z = 0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 0.0

            self.marker_pub.publish(marker)

            end_time = time.time()
            if len(tags):
                # print('EA',e_angl_deg)
                rospy.loginfo(f'Car detected, time: {end_time - start_time}.\nAprilTag x:{AT_POINT_OF_INTEREST[0]},y:{AT_POINT_OF_INTEREST[1]},rot:{-e_angl_deg[1]}\nOur method x:{POINT_OF_INTEREST[0]},y:{POINT_OF_INTEREST[1]}')
            else:
                rospy.logwarn('No apriltags')


    def get_grad(self, data):

        kernel_x = np.array([[-1, 1]])
        kernel_y = np.array([[-1], [1]])

        grad_x = cv2.filter2D(data, -1, kernel_x)
        grad_y = cv2.filter2D(data, -1, kernel_y)

        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        return magnitude
    
    def get_centroid_it(self, points):
        x = points[0]
        y = points[1]

        num_it = 1                     # number of iterations
        #radius = self.detect_r          # circle radius
        w = self.rec_w                  # rectangle parameters
        h = self.rec_h
        # w = self.init_rw                # adaptive rectangle 
        # h = self.init_rh
        # points = points.copy()

        if len(x) == 0:
            rospy.logwarn('No points')
            return None,None

        for i in range(num_it):
            centroid_x = np.mean(x)
            centroid_y = np.mean(y)

            if not np.isnan(centroid_x) and not np.isnan(centroid_y):
                centroid_x = int(centroid_x)
                centroid_y = int(centroid_y)
            else:
                rospy.logwarn('NaN centroid')
                return None, None

            if i == num_it:
                continue
            
            # Compute the squared distance from each point to the centroid - CIRCLE
            # distance_squared = (x - centroid_x)**2 + (y - centroid_y)**2
            # Mask the arrays based on the distance from the centroid
            # x = x[distance_squared <= radius**2]
            # y = y[distance_squared <= radius**2]

            # RECTANGLE
            left_boundary = centroid_x - w / 2
            right_boundary = centroid_x + w / 2
            top_boundary = centroid_y - h / 2
            bottom_boundary = centroid_y + h / 2

            # Find the indexes of the points that are completely within the rectangle
            inside_rectangle_indexes = np.where((x < left_boundary) & (x > right_boundary) &
                                                (y < top_boundary) & (y > bottom_boundary))[0]

            # Remove the points that are not completely within the rectangle
            x = np.delete(x, inside_rectangle_indexes)
            y = np.delete(y, inside_rectangle_indexes)

            # FIXME: adaptive w,h
            # h -= 5
            # w -= 5

        return centroid_x,centroid_y
    
    def interpolate(self,depth_data):
        zero_indices = (depth_data == 0)

        mask = ~zero_indices

        x, y = np.meshgrid(np.arange(depth_data.shape[1]), np.arange(depth_data.shape[0]))

        interpolated_values = np.zeros_like(depth_data)
        interpolated_values[zero_indices] = 0
        for i in range(depth_data.shape[0]):
            if np.any(mask[i]):
                interpolated_values[i] = np.interp(x[i], x[i][mask[i]], depth_data[i][mask[i]])
            else:
                return depth_data
        return interpolated_values    

    def process_scan(self,scan:LaserScan):
        """
        input: LaserScan 
        output: array of (x,y) points in Laser coordinate frame
        """
        start_angle = scan.angle_min
        end_angle = scan.angle_max
        a_inc = scan.angle_increment
        ranges = scan.ranges
        max_rng = scan.range_max
        min_rng = scan.range_min

        coords = []
        c_angle = start_angle
        for r in ranges:
            if not self.__is_valid(max_rng,min_rng,r):
                c_angle += a_inc
                continue
            x = r*np.cos(c_angle)
            y = r*np.sin(c_angle)
            coords.append((x,y))
            c_angle+=a_inc

        return coords

    def scancb(self,scan):
      self.scan = scan
    
    def __is_valid(self,max_r,min_r,r):
        """
        checks if the range measurement is valid
        """
        in_range = True
        if max_r < r or r < min_r:
            in_range = False
        return in_range


    def pix_to_camera(self,pixel,depth=1):
        u = pixel[1]
        v = pixel[0]

        # print('P',self.P)
        fx = self.P[0,0]
        fy = self.P[1,1]
        cx = self.P[0,2]
        cy = self.P[1,2]

        # x = (u-cx)*depth/fx
        # y = -(v-cy)*depth/fy

        x = depth/1000
        y = -(u-cx)*(depth/1000)/fx

        # x = (u-self.K[0,2])/self.K[0,0]
        # y = (v-self.K[1,2])/self.K[1,1]

        # scale = 1
        # uvw = scale*np.array([u,v,1]).T
        # P_inv = np.linalg.pinv(self.P)
        # dir_cam = P_inv@uvw

        # dir_cam /= np.linalg.norm(dir_cam)*depth

        # # print('dc',dir_cam,dir_cam.shape)
        # dir_cam = dir_cam[:3]

        # dir_wld = self.R@dir_cam

        point = [x,y]

        # sin_r = np.sin(15*np.pi/180)
        # cos_r = np.cos(15*np.pi/180)
        # Rot = np.array([[cos_r,-sin_r],[sin_r, cos_r]])

        # hom_p = np.array([x,y,1]).T
        # point = Rot@hom_p
        # print(point)
        # rospy.loginfo(f'point {point}')

        return point
    
    def bfs_filter(self,depth,image,d0,c):

        if d0 == 1:
            return image
        
        # filtered = np.copy(image)
        mask = np.zeros_like(depth)
        visited = np.zeros_like(depth)

        chunk = 10

        surr = [(-chunk,0),(chunk,0),(0,chunk),(0,-chunk),(chunk,chunk),(-chunk,-chunk),(-chunk,chunk),(chunk,-chunk)]
        # print(surr)

        q = [c]
        # mask[c[0],c[1]] = 1
        # FIXME check for boundaries
        mask[c[0]-chunk:c[0]+chunk,c[1]-chunk:c[1]+chunk] = 1
        visited[c[0],c[1]] = 1     

        dist_d = 250
        dist_c = 100
        # print('start')
        while len(q):
            # print('len',len(q))
            u = np.array(q.pop(0))
            q_depth = depth[c[0],c[1]]
            q_col = image[c[0],c[1]]

            for s in surr:
                v = u+s
                if 0<=v[0]<depth.shape[0] and 0<=v[1]<depth.shape[1]:
                    tmp_d = depth[v[0],v[1]]
                    tmp_col = image[c[0],c[1]]
                    # print('tmp_d',tmp_d)
                    if abs(tmp_d-q_depth) <= dist_d and self.__rgb_close(q_col,tmp_col)<=dist_c:
                        # print('vis',visited[v[0],v[1]])
                        if visited[v[0],v[1]] == 0:
                            q.append(v)
                            mask[v[0]-chunk:v[0]+chunk,v[1]-chunk:v[1]+chunk] = 1
                            visited[v[0],v[1]] = 1

        # print('end')
        # print('im,d',image.shape,mask.shape)
        # filtered = cv2.bitwise_and(image,image,mask=mask)
        # filtered = np.zeros_like(image)
        # filtered[mask] = image[mask]
        mask = np.expand_dims(mask, axis=2)

        # Apply the mask to the image
        filtered = image * mask

        # plt.imshow(mask)
        # plt.show()

        return filtered

    def __rotation_matrix_to_euler_angles(self,R):
        # Assuming the angles are in radians.
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])

    def __euler_to_quaternion(self,roll, pitch, yaw):
        # quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        # quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        quaternion = tf.transformations.quaternion_from_euler(0, 0, -pitch)
        return quaternion
    
    def __rgb_close(self,p1,p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))



if __name__ == '__main__':
    print('DEMO CAMERA SUBSCRIBER')
    rospy.init_node('Image_node',anonymous=True)
    ProcessData(visualise=True)
    rospy.spin()
