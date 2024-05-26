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
from scipy.interpolate import griddata
from scipy.ndimage import median_filter

class ProcessImage:

    def __init__(self,visualise=False) -> None:
        rospy.loginfo('Initializing node')
        self.initialized = False

        self.im_sub = message_filters.Subscriber('/camera/color/image_raw',Image)
        self.depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw',Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.im_sub, self.depth_sub], queue_size=10, slop=0.5)
        self.ts.registerCallback(self.process_camera)

        self.vis = visualise
        self.brige = CvBridge()

        self.detect_r = 100 # circle radius
        self.init_rw = 300
        self.init_rh = 150
        self.rec_w = 200 # rec width
        self.rec_h = 100 # rec height

        self.initialized = True
        rospy.loginfo('Node initialized')

    def process_camera(self,img:Image,depth:Image):
        '''
        Process the data
        '''
        img_data = self.brige.imgmsg_to_cv2(img,desired_encoding="bgr8")            # converts to image (rgb8), to visualise with cv2 in true color use bgr8
        depth_data = self.brige.imgmsg_to_cv2(depth,desired_encoding="passthrough") # 16UC1

        gray_img = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)                       # grayscale image
        if self.vis:
            hsv_im = cv2.cvtColor(img_data,cv2.COLOR_BGR2HSV)                           # hsv image - not used now

        blurred_depth = cv2.GaussianBlur(depth_data, (13, 13), 1)                   # depth filtered with gaussian 
        grad_blurred = self.get_grad(blurred_depth)                                 # magnitude of x,y gradients
        cv2.imshow('blurred depth',grad_blurred)

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
        cv2.imshow('Filtered depth',grad_filt_b)

        interest_mask = np.ones_like(depth_data)                                    # filters out the region of interest
        interest_mask[:50,:] = 0                
        interest_mask[380:,:] = 0

        merged_grad = cv2.bitwise_and(grad_filt_b,grad_im_filt.astype(np.float32),mask=interest_mask.astype(np.uint8)) # merges the gradients together

        # VISUALISE DEPTH:
        if self.vis:
            rgb_from_hsv = cv2.cvtColor(hsv_im,cv2.COLOR_HSV2BGR)
            rgb_from_hsv = cv2.bitwise_and(rgb_from_hsv,rgb_from_hsv,mask=interest_mask.astype(np.uint8))
            light_green = (144, 238, 144)  
            replace_black_mask = (rgb_from_hsv == [0, 0, 0]).all(axis=2)        # replace black with light green for better visualisation
            rgb_from_hsv[replace_black_mask] = light_green


            normalized_depth_data = (depth_data- np.min(depth_data)) / (np.max(depth_data) - np.min(depth_data))
            # Apply a colormap to the normalized depth data
            depth_colormap = cv2.applyColorMap(np.uint8(normalized_depth_data * 255), cv2.COLORMAP_JET)
            cv2.imshow('int',depth_colormap)
        

            n_grad_im = ((grad_im - np.min(grad_im)) / (np.max(grad_im) - np.min(grad_im)) * 255).astype(np.uint8)
            n_grad_imf = ((grad_im_filt - np.min(grad_im_filt)) / (np.max(grad_im_filt) - np.min(grad_im_filt)) * 255).astype(np.uint8)
            cv2.imshow('gray grad',n_grad_im)
            cv2.imshow('gray grad filt',n_grad_imf)

            hgh2 = rgb_from_hsv.copy()
            hgh2[merged_grad==1] = (0,0,255)
        
        if self.vis:
            mg_vis = merged_grad.astype(np.uint8) * 255      

        centroid_row,centroid_col = self.get_centroid_it(np.where(merged_grad==1))  # gets the centroid of the points                         

        if centroid_row != None:
            cr = 2          # set the  width for the mean
            depth_mean = np.mean(depth_data[centroid_row-cr:centroid_row+cr,centroid_col-cr:centroid_col+cr])   # get the mean of depth in the centroid

            if self.vis:
                hgh2[centroid_row-5:centroid_row+5,centroid_col-5:centroid_col+5] = (255,0,255) 

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

            if self.vis:
                tl = (int(centroid_col-rec_w/2),int(centroid_row-rec_h/2)) # top left
                br = (int(centroid_col+rec_w/2),int(centroid_row+rec_h/2)) # bottom right
                cv2.circle(image_final, (centroid_col,centroid_row), self.detect_r, (255, 255, 255), 1)
                cv2.rectangle(image_final, tl, br, (0,255,0), thickness=2)

                mg_vis3 = np.repeat(mg_vis[:, :, np.newaxis], 3, axis=2) # creating 3d array for visualisation purpose only
                cv2.rectangle(mg_vis3, tl, br, (0,255,0), thickness=2)

                cv2.imshow('merged grad',mg_vis3)
                cv2.imshow('final detection',image_final)
            

        if self.vis:
            # VISUAlISATION

            # cv2.imshow('d',depth_data)
            # cv2.imshow('dc',depth_colormap)
            # cv2.imshow('original',img_data)
            # cv2.imshow('g',gradient)
            # cv2.imshow('gi',grad_filt)
            # cv2.imshow('gb',grad_filt_b)
            # cv2.imshow('im',filt_im)
            # cv2.imshow('zers',mask)
            # cv2.imshow('highlight',rgb_from_hsv)
            # cv2.imshow('canny',edge)
            # cv2.imshow('inv',gf2)
            # cv2.imshow('gray',gray_img)
            # cv2.imshow('gray grad',n_grad_im)
            # cv2.imshow('gray grad filt',n_grad_imf)
            # cv2.imshow('merged grad',mg_vis)
            # cv2.imshow('highlight2',hgh2)
            # cv2.imshow('zers interp',mask2)
            # cv2.imshow('filt int grad',grad_filt_int)
            # cv2.imshow('secder',grad_filt_sd)
            
            cv2.waitKey(1)

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

        num_it = 10                     # number of iterations
        radius = self.detect_r          # circle radius
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
            x = np.delete(x, ~inside_rectangle_indexes)
            y = np.delete(y, ~inside_rectangle_indexes)

            # FIXME: adaptive w,h
            # h -= 5
            # w -= 5

        return centroid_x,centroid_y


if __name__ == '__main__':
    print('DEMO CAMERA SUBSCRIBER')
    rospy.init_node('Image_node',anonymous=True)
    ProcessImage(visualise=True)
    rospy.spin()
