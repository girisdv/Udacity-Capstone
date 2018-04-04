from styx_msgs.msg import TrafficLight
import numpy as np

import rospy
import sys
import cv2

class TLClassifier(object):
    def __init__(self):
        pass
        #TODO load classifier

    def check_specific_color_tl(self, image, color_test1, color_test2):
        # using Hue/Saturation/Value (HSV) color space
        # https://en.wikipedia.org/wiki/HSL_and_HSV
        # works well for base colors (red/blue/green)
        # https://docs.opencv.org/3.1.0/da/d53/tutorial_py_houghcircles.html
        # http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
        

        # constants
        h_delta = 5
        sv_low = 50
        sv_high = 255
        add_weight = 1.0
        extra_weight = 0.0
        gb_kernel = (11,11)
        
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # color 1 range
        lbound = np.array([(color_test1-h_delta), sv_low,sv_low])
        hbound = np.array([(color_test1+h_delta), sv_high,sv_high])
        hsv_mask1 = cv2.inRange(image_hsv, lbound, hbound)

        # color 2 range
        lbound = np.array([(color_test2-h_delta), sv_low,sv_low])
        hbound = np.array([(color_test2+h_delta), sv_high,sv_high])
        hsv_mask2 = cv2.inRange(image_hsv, lbound, hbound)

        # combine
        hsv_combined = cv2.addWeighted(hsv_mask1, add_weight, hsv_mask2, add_weight, extra_weight)

        # blur with large kernel
        hsv_bl = cv2.GaussianBlur(hsv_combined, gb_kernel, 0)

        # hough gradient, find circles separated by enough distance
        find_circles = cv2.HoughCircles(hsv_bl, cv2.HOUGH_GRADIENT,
                                        .5, 40, param1 = 70, param2 = 30,
                                        minRadius = 5, maxRadius = 100)
        return (find_circles is not None)
    

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction

        # for red use bright red and magenta red
        # base red is [0,255,255] in HSV, using 5, magenta is 175
        if (self.check_specific_color_tl(image, 5, 175)):
            return TrafficLight.RED
        # green seems to work with base green, [60,255,255] in HSV
        elif (self.check_specific_color_tl(image, 60, 60)):
            return TrafficLight.GREEN

        return TrafficLight.UNKNOWN
