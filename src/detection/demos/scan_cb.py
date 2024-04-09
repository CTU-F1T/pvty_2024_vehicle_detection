"""
    Simple scan subscrber. Clusters the points - WRONG, needs to be rewritten.
"""

import rospy
from sensor_msgs.msg import LaserScan
import numpy as np

# import matplotlib
# matplotlib.use('Qt5Agg')  # Use the 'agg' backend
import matplotlib.pyplot as plt
# plt.switch_backend('tkagg')
import time

class ProcessScan():

    def __init__(self,visualise=False) -> None:
        rospy.loginfo('Initializing node')
        self.initialized = False

        rospy.Subscriber('/scan',LaserScan,self.scan_cb)
        # rospy.Subscriber('/scan_dummy',LaserScan,self.scan_cb)

        self.clusters = []
        self.tolerance = 0.05
        self.vis = visualise

        if self.vis:
            # plt.figure()
            plt.ion()  # Enable interactive mode
        #     self.fig, self.ax = plt.subplots()
        #     self.ax.set_aspect('equal')
        #     plt.show()

        self.initialized = True
 
        rospy.loginfo('Process scan node initialized. Waiting for data.')

    def __is_valid(self,max_r,min_r,r):
        in_range = True
        if max_r < r or r < min_r:
            print('invalid meas')
            in_range = False
        return in_range


    def scan_cb(self,scan:LaserScan):
        start_angle = scan.angle_min
        end_angle = scan.angle_max
        a_inc = scan.angle_increment
        ranges = scan.ranges
        max_rng = scan.range_max
        min_rng = scan.range_min

        self.clusters = []
        tmp_cluster = []

        c_angle = start_angle
        for rng in ranges:
            if not self.__is_valid(max_rng,min_rng,rng):
                c_angle += a_inc
                continue
            if len(tmp_cluster) == 0:
                tmp_cluster.append((rng,c_angle))
                prev_rng = rng
            else:
                if abs(prev_rng-rng) <= self.tolerance:
                    tmp_cluster.append((rng,c_angle))


                else:
                    # print('append cluster')
                    self.clusters.append(tmp_cluster)
                    tmp_cluster = []
            c_angle += a_inc
        
        grad = np.gradient(ranges)
        dif = np.diff(ranges)

        plt.clf()
        plt.plot(grad,label='Gradient')
        plt.plot(dif,label='Diff')
        plt.plot(ranges,label='ranges')
        plt.legend()
        plt.show()


        if self.vis:
            print('clusters = ',len(self.clusters))
            # plt.close()
            self.visualise(self.clusters,grad)
            # plt.show()

    def pol2car(self,coords):
        cartesian = []
        x_points = []
        y_points = []

        for coord in coords:
            r,angle = coord
            x = -r*np.sin(angle)
            y = r*np.cos(angle)
            cartesian.append((x,y))
            x_points.append(x)
            y_points.append(y)

        return cartesian,x_points,y_points

    def visualise(self,clusters,grad):
        # plt.switch_backend('agg')
        # plt.clf()
        # plt.figure()
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        for cluster in clusters:
            cluster_cart,x_p,y_p = self.pol2car(cluster)
            x_np = np.array(x_p)
            y_np = np.array(y_p)
            # self.ax.plot(x_np, y_np)
            # plt.plot(x_np,y_np)
            axs[0].plot(x_np, y_np)
        
        axs[1].plot(grad)
        plt.tight_layout()
        plt.show()
        # plt.show()
        # time.sleep(1)
        # plt.pause(0.1)
        # plt.close()
            
        # plt.show()
        # plt.sleep(0.1)
        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()
        # plt.close()
        




if __name__ == '__main__':
    print('Scan test init')
    # ProcessScan(visualise=True)
    ProcessScan(visualise=False)
    rospy.spin()


