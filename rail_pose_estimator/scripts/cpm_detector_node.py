#!/usr/bin/env python
# This file is responsible for bridging ROS to the ObjectDetector class (built with PyCaffe)

from __future__ import division

import sys

import cv2
import rospy
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
from rail_pose_estimation_msgs.msg import Keypoints, Poses

import pose_estimation

# Debug Helpers
FAIL_COLOR = '\033[91m'
ENDC_COLOR = '\033[0m'


def eprint(error):
    sys.stderr.write(
        FAIL_COLOR
        + type(error).__name__
        + ": "
        + error.message
        + ENDC_COLOR
    )
# End Debug Helpers


class CPM(object):
    """
    This class takes in image data and finds / annotates objects within the image
    """

    def __init__(self):
        rospy.init_node('rail_pose_estimator_node')
        self.person_keypoints = []
        self.keypoint_arrays = []
        self.image_datastream = None
        self.input_image = None
        self.bridge = CvBridge()
        self.detector = pose_estimation.PoseMachine()
        self.debug = rospy.get_param('~debug', default=False)
        self.image_sub_topic_name = rospy.get_param('~image_sub_topic_name', default='/kinect/qhd/image_color_rect')
        self.use_compressed_image = rospy.get_param('~use_compressed_image', default=False)
        self.part_str = ['nose', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'left_shoulder', 'left_elbow',
                      'left_wrist', 'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'right_eye', 'left_eye', 'right_ear',
                      'left_ear']

    def _convert_msg_to_image(self, image_msg):
        """
        Convert an incoming image message (compressed or otherwise) into a cv2
        image
        """
        if not self.use_compressed_image:
            try:
                image_cv = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            except CvBridgeError as e:
                print e
                return None
        else:
            image_np = np.fromstring(image_msg.data, np.uint8)
            image_cv = cv2.imdecode(image_np, cv2.CV_LOAD_IMAGE_COLOR)

        return image_cv

    def _parse_image(self, image_msg):

        header = image_msg.header
        image_cv = self._convert_msg_to_image(image_msg)
        if image_cv is None:
            return
        # self.person_keypoints = self.detector.estimate_keypoints(image_cv)
        # candidate, subset = self.detector.estimate_keypoints(image_cv)
        people = self.detector.estimate_keypoints(image_cv)
        #### DEBUG ####
        if self.debug:
            # out_image = self.detector.visualize_keypoints(image_cv, candidate, subset)
            out_image = self.detector.visualize_keypoints(image_cv, people)
            try:
                image_msg = self.bridge.cv2_to_imgmsg(out_image, "bgr8")
            except CvBridgeError as e:
                print e

            image_msg.header = header
            self.image_pub.publish(image_msg)
        #### END DEBUG ####

        # Instantiate poses object
        obj_arr = Poses()
        obj_arr.header = header
        for person in people:
            msg = Keypoints()
            nose = person.get('nose', [-1, -1])
            msg.nose_y = nose[0]
            msg.nose_x = nose[1]

            neck = person.get('neck', [-1, -1])
            msg.neck_y = neck[0]
            msg.neck_x = neck[1]

            right_shoulder = person.get('right_shoulder', [-1, -1])
            msg.right_shoulder_y = right_shoulder[0]
            msg.right_shoulder_x = right_shoulder[1]
            left_shoulder = person.get('left_shoulder', [-1, -1])
            msg.left_shoulder_y = left_shoulder[0]
            msg.left_shoulder_x = left_shoulder[1]

            right_elbow = person.get('right_elbow', [-1, -1])
            msg.right_elbow_y = right_elbow[0]
            msg.right_elbow_x = right_elbow[1]
            left_elbow = person.get('left_elbow', [-1, -1])
            msg.left_elbow_y = left_elbow[0]
            msg.left_elbow_x = left_elbow[1]

            right_wrist = person.get('right_wrist', [-1, -1])
            msg.right_wrist_y = right_wrist[0]
            msg.right_wrist_x = right_wrist[1]
            left_wrist = person.get('left_wrist', [-1, -1])
            msg.left_wrist_y = left_wrist[0]
            msg.left_wrist_x = left_wrist[1]

            Lhip = person.get('Lhip', [-1, -1])
            msg.left_hip_y = Lhip[0]
            msg.left_hip_x = Lhip[1]
            Rhip = person.get('Rhip', [-1, -1])
            msg.right_hip_y = Rhip[0]
            msg.right_hip_x = Rhip[1]

            left_eye = person.get('left_eye', [-1, -1])
            msg.left_eye_y = left_eye[0]
            msg.left_eye_x = left_eye[1]
            right_eye = person.get('right_eye', [-1, -1])
            msg.right_eye_y = right_eye[0]
            msg.right_eye_x = right_eye[1]

            right_ear = person.get('right_ear', [-1, -1])
            msg.right_ear_y = right_ear[0]
            msg.right_ear_x = right_ear[1]
            left_ear = person.get('left_ear', [-1, -1])
            msg.left_ear_y = left_ear[0]
            msg.left_ear_x = left_ear[1]

            Rkne = person.get('Rkne', [-1, -1])
            msg.right_knee_y = Rkne[0]
            msg.right_knee_x = Rkne[1]
            Lkne = person.get('Lkne', [-1, -1])
            msg.left_knee_y = Lkne[0]
            msg.left_knee_x = Lkne[1]

            Rank = person.get('Rank', [-1, -1])
            msg.right_ankle_y = Rank[0]
            msg.right_ankle_x = Rank[1]
            Lank = person.get('Lank', [-1, -1])
            msg.left_ankle_y = Lank[0]
            msg.left_ankle_x = Lank[1]

            obj_arr.people.append(msg)

        self.object_pub.publish(obj_arr)

    def run(self,
            pub_image_topic='~debug/poses_image',
            pub_object_topic='~poses'):
        if not self.use_compressed_image:
            rospy.Subscriber(self.image_sub_topic_name, Image, self._parse_image)
        else:
            rospy.Subscriber(self.image_sub_topic_name+'/compressed', CompressedImage, self._parse_image)
        if self.debug:
            self.image_pub = rospy.Publisher(pub_image_topic, Image, queue_size=2) # image publisher
        self.object_pub = rospy.Publisher(pub_object_topic, Poses, queue_size=2) # objects publisher
        rospy.spin()

if __name__ == '__main__':
    try:
        detector = CPM()
        detector.run()
    except rospy.ROSInterruptException:
        pass
