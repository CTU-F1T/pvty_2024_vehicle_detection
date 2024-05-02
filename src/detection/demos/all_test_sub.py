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
from sklearn.cluster import DBSCAN

class ProcessData:  # rename to detect?
    """
    Class to process the data
    """

    def __init__(self,visualise=False) -> None:
        rospy.loginfo('Initializing node')
        self.initialized = False

        # self.im_sub = message_filters.Subscriber('/camera/color/image_raw',Image)
        # self.depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw',Image)
        self.scan_sub = message_filters.Subscriber('/scan_dummy',LaserScan)
        # self.ts = message_filters.ApproximateTimeSynchronizer([self.im_sub, self.depth_sub, self.scan_sub], queue_size=10, slop=0.5)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.scan_sub], queue_size=10, slop=0.5)
        self.ts.registerCallback(self.process_data)

        self.vis = visualise
        self.brige = CvBridge()

        self.initialized = True
        rospy.loginfo('Node initialized')

    # def process_data(self,img:Image,depth:Image,scan:LaserScan):
    def process_data(self,scan:LaserScan):
        """
        input: image: image <Image> from the camera
               depth: depth <Image> from the camera lidar, aligned 
               scan: <LaserScan> from the planar lidar
        """
        
        POINT_OF_INTEREST = [195, 390]

        # img_data = self.brige.imgmsg_to_cv2(img,desired_encoding="passthrough")     # converts to image (rgb8), to visualise with cv2 in true color use bgr8
        # depth_data = self.brige.imgmsg_to_cv2(depth,desired_encoding="passthrough") # 16UC1
        scan_data = self.process_scan(scan)
        scan_data = [point for point in scan_data if not math.isnan(point[0]) and not math.isnan(point[1])]
        
        # perform dbscan
        clustering = DBSCAN(eps=0.1, min_samples=5).fit(scan_data)
        labels = clustering.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)
        
        # map data
        min_x = scan_data[0][0]
        min_y = scan_data[0][1]
        max_x = scan_data[0][0]
        max_y = scan_data[0][1]
        for x, y in scan_data:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        scan_map = np.zeros((500, 500), dtype=np.uint8)
        for x, y in scan_data:
            x = ((-min_x + x) / (max_x - min_x))*499
            y = ((-min_y + y) / (max_y - min_y))*499
            x = round(x)
            y = round(y)
            scan_map[x, y] = 1  # Set RGB values to (255, 255, 255) at the coordinate
            
        # image
        scan_img = np.zeros((500, 500, 3), dtype=np.uint8)
        scan_img[scan_map == 1] = (255, 255, 255)
        
        # dbscan into image
        unique_labels = set(labels)
        # core_samples_mask = np.zeros_like(labels, dtype=bool)
        # core_samples_mask[clustering.core_sample_indices_] = True
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = (255, 0, 0)
            else:
                col = [int(c*255) for c in col[0:3]]

            indexes = [index for index, value in enumerate(labels) if value == k]
            for i in indexes:
                x = scan_data[i][0]
                y = scan_data[i][1]
                x = ((-min_x + x) / (max_x - min_x))*499
                y = ((-min_y + y) / (max_y - min_y))*499
                x = round(x)
                y = round(y)
                scan_img[x, y] = col
            
        scan_img[POINT_OF_INTEREST[0], POINT_OF_INTEREST[1]] = (0, 0, 255)
        cv2.imshow("Image", scan_img)
        cv2.waitKey(0)

        return
    
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
    ProcessData(visualise=False)
    rospy.spin()
