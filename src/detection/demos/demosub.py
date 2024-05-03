import rospy
from sensor_msgs.msg import Image,LaserScan
import message_filters


class DemoSub:
  def __init__(self):
    rospy.loginfo('Starting node')
    self.im_sub = message_filters.Subscriber('camera/color/image_raw',Image)
    self.depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw',Image)
    #self.scan_sub = message_filters.Subscriber('/scan',LaserScan)
    #self.ts = message_filters.TimeSynchronizer([self.im_sub,self.depth_sub,self.scan_sub],10)
    self.ts = message_filters.ApproximateTimeSynchronizer([self.im_sub,self.depth_sub],queue_size=10,slop=0.5)
    self.ts.registerCallback(self.demosub)
    self.scan = None
    self.scan_sub = rospy.Subscriber('/scan',LaserScan,self.scancb)

    rospy.loginfo('Init done')

  def demosub(self, img, depth):
    #rospy.loginfo('callback')
    if self.scan is not None:
      print('timestamps:')
      print('img',img.header.stamp.to_sec())
      print('dep',depth.header.stamp.to_sec())
      print('scn',self.scan.header.stamp.to_sec())
      print('diff',(self.scan.header.stamp.to_sec()-img.header.stamp.to_sec()))

  def scancb(self,scan):
    self.scan = scan

if __name__ == '__main__':
  print('start')
  rospy.init_node('demoscan',anonymous=True)
  DemoSub()
  rospy.spin()
