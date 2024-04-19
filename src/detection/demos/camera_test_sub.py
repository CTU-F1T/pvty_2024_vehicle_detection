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

        self.detect_r = 100
        self.init_rw = 300
        self.init_rh = 150
        self.rec_w = 200 # rec width
        self.rec_h = 100 # rec height

        self.initialized = True

    def process_camera(self,img:Image,depth:Image):
        img_data = self.brige.imgmsg_to_cv2(img,desired_encoding="bgr8")     # converts to image (rgb8), to visualise with cv2 in true color use bgr8
        depth_data = self.brige.imgmsg_to_cv2(depth,desired_encoding="passthrough") # 16UC1

        gray_img = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
        # if self.vis:
        #     cv2.imshow('image',img_data)
            # cv2.waitKey(1)

        # interp = self.interpolate(depth_data)
        

        hsv_im = cv2.cvtColor(img_data,cv2.COLOR_BGR2HSV)
        # plt.imshow(hsv_im)
        # plt.show()
        # hsv_low = np.array([0,120,80])
        # hsv_high = np.array([100,230,120])
        # hsv_mask = cv2.inRange(hsv_im,hsv_low,hsv_high)
        # hsv_filtered = cv2.bitwise_and(hsv_im,hsv_im,mask=~hsv_mask)
        # if self.vis:
            # cv2.imshow('hsv',hsv_im)
            # cv2.imshow('hsv filt',hsv_filtered)
            # cv2.waitKey(1)


        blurred_depth = cv2.GaussianBlur(depth_data, (13, 13), 1)                   # probably does not help at all

        gradient = self.get_grad(depth_data)                                        # get magnitude of gradients in x and y
        grad_blurred = self.get_grad(blurred_depth)

        th = 10
        th_up = 100000
        th_idx = np.where((gradient>th) & (gradient<th_up))
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

        mask = (depth_data == 0).astype(np.uint8) * 255 #93, 413
        

        interest_mask = np.ones_like(depth_data)
        interest_mask[:50,:] = 0
        interest_mask[380:,:] = 0

        # rgb_from_hsv = cv2.cvtColor(hsv_filtered,cv2.COLOR_HSV2BGR)
        rgb_from_hsv = cv2.cvtColor(hsv_im,cv2.COLOR_HSV2BGR)


        rgb_from_hsv = cv2.bitwise_and(rgb_from_hsv,rgb_from_hsv,mask=interest_mask.astype(np.uint8))
        black_mask = (rgb_from_hsv == [0, 0, 0]).all(axis=2)

        # Create a light green color
        light_green = (144, 238, 144)  # You can adjust this color as needed

        # Replace pixels in the original image with light green where the mask is True
        rgb_from_hsv[black_mask] = light_green
        
        hgh2 = rgb_from_hsv.copy()
        # print(type(rgb_from_hsv))
        # print(rgb_from_hsv.shape)
        rgb_from_hsv[grad_filt==1] = (0,0,255)
        
        # rgb_from_hsv[rgb_from_hsv== [0, 0, 0]).all(axis=2) = (144, 238, 144)
        


        # int_mask_idx = np.where(interest_mask==1)
        # print('bef')
        # depth_data_interp = self.interpolate(depth_data,interest_mask)
        # print('aft')
        # depth_data_interp = self.interpolate(depth_data,np.ones_like(depth_data))
        # mask2 = (depth_data_interp == 0).astype(np.uint8) * 255 #93, 413

        # print('interest mask shape',interest_mask.shape)
        # print('depth shape',depth_data.shape)
        # print('mask',mask.shape)


        normalized_depth_data = (depth_data- np.min(depth_data)) / (np.max(depth_data) - np.min(depth_data))

        # Apply a colormap to the normalized depth data
        depth_colormap = cv2.applyColorMap(np.uint8(normalized_depth_data * 255), cv2.COLORMAP_JET)

        # cv2.imshow('int',depth_colormap)
        # cv2.waitKey(1)
        # plt.imshow(depth_data_interp)
        # plt.show()
        # im_int =  cv2.bitwise_and(depth_data,depth_data,mask=interest_mask)
        # cv2.imshow('int',depth_colormap)
        # cv2.waitKey(1)
        # int_grad = self.get_grad(depth_data_interp)
        # th = 20
        # th_idx_int = int_grad>th
        # grad_filt_int = np.zeros_like(int_grad)
        # grad_filt_int[th_idx_int] = 1

        # cv2.imshow('filt int grad',grad_filt_int)
        # cv2.waitKey(1)


        # plt.imshow(mask)
        # plt.show()

        # secder = self.get_grad(gradient)
        # th_sdidx = secder>th
        # grad_filt_sd = np.zeros_like(secder)
        # grad_filt_sd[th_sdidx] = 1

        # gradient_i = self.get_grad(interp)
        # gf2 = not grad_filt
        # print(grad_filt)
        gf2 = np.ones_like(grad_filt)
        gf2[np.where(grad_filt==1)] = 0

        edge = cv2.Canny(np.uint8(grad_filt), 0, 3,apertureSize=5,L2gradient=True)         
        # edge = median_filter(grad_filt,2)


        th_gi = 8
        gray_img = cv2.GaussianBlur(gray_img, (13, 13), 1)
        grad_im = self.get_grad(gray_img)

        th_idx_gim = grad_im>th_gi
        grad_im_filt= np.zeros_like(grad_im)
        grad_im_filt[th_idx_gim] = 1

        n_grad_im = ((grad_im - np.min(grad_im)) / (np.max(grad_im) - np.min(grad_im)) * 255).astype(np.uint8)
        n_grad_imf = ((grad_im_filt - np.min(grad_im_filt)) / (np.max(grad_im_filt) - np.min(grad_im_filt)) * 255).astype(np.uint8)
        
        # merged_grad = np.logical_and(grad_filt,grad_im_filt).astype(np.uint8) * 255
        # print('t1',type(grad_filt))
        # print('t2',type(grad_im_filt))
        # print('sh1',grad_filt_b.shape)
        # print('sh2',grad_im_filt.shape)
        # print('msk',interest_mask.shape)
        merged_grad = cv2.bitwise_and(grad_filt_b,grad_im_filt.astype(np.float32),mask=interest_mask.astype(np.uint8))
        mg_vis = merged_grad.astype(np.uint8) * 255
        
        hgh2[merged_grad==1] = (0,0,255)

        # rows, cols = np.where(merged_grad == 1)
        #     # Calculate the centroid
        # centroid_row = int(np.mean(rows))
        # centroid_col = int(np.mean(cols))
        # print(centroid_col,centroid_row)
        # hgh2[centroid_row-5:centroid_row+5,centroid_col-5:centroid_col+5] = (255,0,0)

        centroid_row,centroid_col = self.get_centroid_it(np.where(merged_grad==1))

        if centroid_row != None:
            #adaptive rectangle:
            cr = 2
            depth_mean = np.mean(depth_data[centroid_row-cr:centroid_row+cr,centroid_col-cr:centroid_col+cr])
            # self.rec_w = self.init_rw*(depth_mean/1000)
            # self.rec_h = self.init_rh*(depth_mean/1000)

            hgh2[centroid_row-5:centroid_row+5,centroid_col-5:centroid_col+5] = (255,0,255)   
            depth_mean = np.mean(depth_data[centroid_row-cr:centroid_row+cr,centroid_col-cr:centroid_col+cr])
            # print('dm',depth_mean)
            d_margin_l = 100
            d_margin_h = 400
            depth_mask = np.where((depth_data<depth_mean+d_margin_h)&(depth_data>depth_mean-d_margin_l))
            # image_final = cv2.bitwise_and(img_data,img_data,mask=depth_mask.astype(np.uint8))
            image_final = np.zeros_like(img_data)
            image_final[depth_mask] = img_data[depth_mask]

            rec_w = self.rec_w # rec width
            rec_h = self.rec_h # rec height

            tl = (int(centroid_col-rec_w/2),int(centroid_row-rec_h/2)) # top left
            br = (int(centroid_col+rec_w/2),int(centroid_row+rec_h/2)) # bottom right
            cv2.circle(image_final, (centroid_col,centroid_row), self.detect_r, (255, 255, 255), 1)
            cv2.rectangle(image_final, tl, br, (0,255,0), thickness=2)

            # print(mg_vis.shape)
            mg_vis3 = np.repeat(mg_vis[:, :, np.newaxis], 3, axis=2) # creating 3d array for visualisation purpose only
            cv2.rectangle(mg_vis3, tl, br, (0,255,0), thickness=2)

            # if self.vis:
            cv2.imshow('merged grad',mg_vis3)
            cv2.imshow('final detection',image_final)
            cv2.waitKey(1)

        # print('dg',np.max(grad_filt))
        # print('ig',np.max(grad_im_filt))

        # print(type(grad_im),grad_im)
        # print(type(gradient))

        if self.vis:
            # cv2.imshow('d',depth_data)
            # cv2.imshow('dc',depth_colormap)
            cv2.imshow('original',img_data)
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
            cv2.imshow('merged grad',mg_vis)
            cv2.imshow('highlight2',hgh2)
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

        # if self.vis:
        #     cv2.imshow('orig im', data)
        #     cv2.imshow('grad x', np.abs(grad_x).astype(np.uint8))
        #     cv2.imshow('grad y', np.abs(grad_y).astype(np.uint8))
        #     cv2.imshow('grad magnitude', magnitude.astype(np.uint8))
            # cv2.waitKey(1)

        return magnitude
    
    def interpolate(self,depth_data, mask_interest):
        # Create grid coordinates
        y, x = np.indices(depth_data.shape)

        # Find coordinates of known depth values (non-zero) and their corresponding values
        known_depth_coords = np.column_stack((x[mask_interest == 1], y[mask_interest == 1]))
        known_depth_values = depth_data[mask_interest == 1]

        # Interpolate unknown depth values (zero) using griddata
        interpolated_values = griddata(known_depth_coords, known_depth_values, (x, y), method='linear')

        # Replace NaN values (outside the convex hull of the known points) with zeros
        interpolated_values[np.isnan(interpolated_values)] = 0

        return interpolated_values    
    
    def get_centroid_it(self, points):
        x = points[0]
        y = points[1]

        num_it = 10
        # radius = 150  # Adjust as needed
        radius = self.detect_r
        w = self.rec_w
        h = self.rec_h
        # w = self.init_rw
        # h = self.init_rh
        # points = points.copy()
        # print(points)
        if len(x) == 0:
            rospy.logwarn('No points')
            return None,None

        for i in range(num_it):
            centroid_x = np.mean(x)
            centroid_y = np.mean(y)

            if not np.isnan(centroid_x) and not np.isnan(centroid_y):
                # print('converted to int')
                centroid_x = int(centroid_x)
                centroid_y = int(centroid_y)
            else:
                # print('rip')
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
            inside_rectangle_indexes = np.where((x >= left_boundary) & (x <= right_boundary) &
                                                (y >= top_boundary) & (y <= bottom_boundary))[0]

            # Remove the points that are completely within the rectangle
            x = np.delete(x, inside_rectangle_indexes)
            y = np.delete(y, inside_rectangle_indexes)

            # FIXME: adaptive w,h
            # h -= 5
            # w -= 5

        # print('returning',centroid_x,centroid_y)
        return centroid_x,centroid_y
        # for i in range(num_it):
        #     rows, cols = np.where(points == 1)
        #     # Calculate the centroid
        #     centroid_row = int(np.mean(rows))
        #     centroid_col = int(np.mean(cols))
        #     # remove outliers
        #     # Create a grid of coordinates
        #     if i == num_it:
        #         continue
        #     x, y = np.meshgrid(np.arange(points.shape[1]), np.arange(points.shape[0]))
        #     # Compute the squared distance from each cell to the centroid
        #     distance_squared = (x - centroid_col)**2 + (y - centroid_row)**2
        #     # Mask the array based on the distance from the centroid
        #     points[distance_squared > radius**2] = 0
        
        # return centroid_row,centroid_col
        


if __name__ == '__main__':
    print('DEMO CAMERA SUBSCRIBER')
    rospy.init_node('Image_node',anonymous=True)
    ProcessImage(visualise=False)
    rospy.spin()
