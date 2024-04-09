"""
    Simple subscriber to show and test how to work with camera topics
"""

import rospy
import numpy as np
from sensor_msgs.msg import Image
import matplotlib.pyplot as plt
import cv2
from cv_bridge import CvBridge

class ProcessImage:

    def __init__(self,visualise) -> None:
        rospy.loginfo('Initializing node')
        self.initialized = False

        rospy.Subscriber('/camera/color/image_raw',Image,self.image_cb)
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw',Image,self.depth_cb)

        self.vis = visualise
        self.brige = CvBridge()
        self.img = None         # image data
        self.imgst = None       # image stamp - synchro?
        self.depth = None       # depth data
        self.dephtst = None     # depth stamp - synchronization?


        self.initialized = True


    def image_cb(self,img: Image):
        print('img:',img.height,img.width,img.encoding)
        print('img',img.header.stamp)
        img_data = self.brige.imgmsg_to_cv2(img,desired_encoding="passthrough") # converts to image (rgb8), to visualise with cv2 in true color use bgr8
        cv2.imshow('image',img_data)
        cv2.waitKey(1)
        
    
    def depth_cb(self,depth:Image):
        print('depth',depth.encoding)
        depth_data = self.brige.imgmsg_to_cv2(depth,desired_encoding="passthrough")     # 16UC1
        print(depth_data.shape)
        # cv2.imshow('depth',depth_data)
        # cv2.waitKey(1)
        


if __name__ == '__main__':
    print('DEMO CAMERA SUBSCRIBER')
    rospy.init_node('Image_node',anonymous=True)
    ProcessImage(visualise=False)
    rospy.spin()
