"""
    Simple subscriber to show and test how to work with camera topics
"""

import rospy
import numpy as np
from sensor_msgs.msg import Image
import matplotlib.pyplot as plt
import cv2
from cv_bridge import CvBridge
import message_filters      # for calback data synchronisation
from sensor_msgs.msg import LaserScan

class ProcessData:  # rename to detect?
    """
    Class to process the data
    """

    def __init__(self,visualise=False) -> None:
        rospy.loginfo('Initializing node')
        self.initialized = False

        self.im_sub = message_filters.Subscriber('/camera/color/image_raw',Image)
        self.depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw',Image)
        self.scan_sub = message_filters.Subscriber('/scan',LaserScan)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.im_sub, self.depth_sub, self.scan_sub], queue_size=10, slop=0.5)
        self.ts.registerCallback(self.process_data)

        self.vis = visualise
        self.brige = CvBridge()

        self.initialized = True
        rospy.loginfo('Node initialized')

    def process_data(self,img:Image,depth:Image,scan:LaserScan):
        """
        input: image: image <Image> from the camera
               depth: depth <Image> from the camera lidar, aligned 
               scan: <LaserScan> from the planar lidar
        """

        img_data = self.brige.imgmsg_to_cv2(img,desired_encoding="passthrough")     # converts to image (rgb8), to visualise with cv2 in true color use bgr8
        depth_data = self.brige.imgmsg_to_cv2(depth,desired_encoding="passthrough") # 16UC1
        scan_data = self.process_scan(scan)


        if self.vis:
            cv2.imshow('image',img_data)
            # cv2.waitKey(1)

        # interp = self.interpolate(depth_data)
        
        blurred_depth = cv2.GaussianBlur(depth_data, (13, 13), 1)                   # probably does not help at all

        gradient = self.get_grad(depth_data)                                        # get magnitude of gradients in x and y
        grad_blurred = self.get_grad(blurred_depth)

        th = 100
        th_idx = gradient>th
        grad_filt = np.zeros_like(gradient)
        grad_filt[th_idx] = 1

        th_idx_b = grad_blurred>th
        grad_filt_b= np.zeros_like(grad_blurred)
        grad_filt_b[th_idx_b] = 1

        filt_im = np.zeros_like(img_data)
        filt_im[th_idx] = img_data[th_idx]

        z_id = (depth_data == 0)
        zers = np.zeros_like(depth_data)
        zers[z_id] = 1

        mask = (depth_data == 0).astype(np.uint8) * 255

        # gradient_i = self.get_grad(interp)
        if self.vis:
            cv2.imshow('d',depth_data)
            cv2.imshow('original',img_data)
            cv2.imshow('g',gradient)
            cv2.imshow('gi',grad_filt)
            cv2.imshow('gb',grad_filt_b)
            cv2.imshow('im',filt_im)
            cv2.imshow('zers',mask)
            cv2.waitKey(1)



    def get_grad(self, data):

        kernel_x = np.array([[-1, 1]])
        kernel_y = np.array([[-1], [1]])

        grad_x = cv2.filter2D(data, -1, kernel_x)
        grad_y = cv2.filter2D(data, -1, kernel_y)

        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        if self.vis:
            cv2.imshow('orig im', data)
            cv2.imshow('grad x', np.abs(grad_x).astype(np.uint8))
            cv2.imshow('grad y', np.abs(grad_y).astype(np.uint8))
            cv2.imshow('grad magnitude', magnitude.astype(np.uint8))
            # cv2.waitKey(1)

        return magnitude
    
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
    
    def __is_valid(self,max_r,min_r,r):
        """
        checks if the range measurement is valid
        """
        in_range = True
        if max_r < r or r < min_r:
            in_range = False
        return in_range

if __name__ == '__main__':
    print('DEMO CAMERA SUBSCRIBER')
    rospy.init_node('Image_node',anonymous=True)
    ProcessImage(visualise=False)
    rospy.spin()
