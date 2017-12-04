#!/usr/bin/env python
# This file is responsible for bridging ROS to the ObjectDetector class (built with PyCaffe)

from __future__ import division

import sys

import cv2
import rospy
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
from rail_pose_estimator.msg import Keypoints, Poses

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
            for part in self.part_str:
                # create a dictionary of part_y part_x and initialize the message with keyword init?
            nose = person.get('nose', [-1, -1])
            nose_y = nose[0]
            nose_x = nose[1]
            neck = person.get('neck', [-1, -1])
            neck_y = neck[0]
            neck_x = neck[1]
            right_shoulder = person.get('right_shoulder', [-1, -1])
            right_shoulder_y = right_shoulder[0]
            right_shoulder_x = right_shoulder[1]
            left_shoulder = person.get('left_shoulder', [-1, -1])
            left_shoulder_y = left_shoulder[0]
            left_shoulder_x = left_shoulder[1]
            right_elbow = person.get('right_elbow', [-1, -1])
            right_elbow_y = right_elbow[0]
            right_elbow_x = right_elbow[1]
            left_elbow = person.get('left_elbow', [-1, -1])
            left_elbow_y = left_elbow[0]
            left_elbow_x = left_elbow[1]
            right_wrist = person.get('right_wrist', [-1, -1])
            right_wrist_y = right_wrist[0]
            right_wrist_x = right_wrist[1]
            left_wrist = person.get('left_wrist', [-1, -1])
            left_wrist_y = left_wrist[0]
            left_wrist_x = left_wrist[1]
            left_eye = person.get('left_eye', [-1, -1])
            left_eye_y = left_eye[0]
            left_eye_x = left_eye[1]
            right_eye = person.get('right_eye', [-1, -1])
            right_eye_y = right_eye[0]
            right_eye_x = right_eye[1]
            left_eye = person.get('left_eye', [-1, -1])
            left_eye_y = left_eye[0]
            left_eye_x = left_eye[1]
            right_eye = person.get('right_eye', [-1, -1])
            right_eye_y = right_eye[0]
            right_eye_x = right_eye[1]

            msg.eye_vec_x = self.vec_between(left_eye_x, right_eye_x)
            msg.eye_vec_y = self.vec_between(left_eye_y, right_eye_y)

            nose_vec_x = self.vec_between(nose_x, neck_x)
            nose_vec_y = self.vec_between(nose_y, neck_y)
            msg.nose_vec_x = nose_vec_x
            msg.nose_vec_y = nose_vec_y

            right_eye_vec_x = self.vec_between(right_eye_x, neck_x)
            right_eye_vec_y = self.vec_between(right_eye_y, neck_y)
            left_eye_vec_x = self.vec_between(left_eye_x, neck_x)
            left_eye_vec_y = self.vec_between(left_eye_y, neck_y)
            msg.right_eye_angle = self.angle_between([right_eye_vec_x, right_eye_vec_y], [nose_vec_x, nose_vec_y])
            msg.left_eye_angle = self.angle_between([left_eye_vec_x, left_eye_vec_y], [nose_vec_x, nose_vec_y])

            left_elbow_vec_x = self.vec_between(left_elbow_x, left_shoulder_x)
            left_elbow_vec_y = self.vec_between(left_elbow_y, left_shoulder_y)
            left_to_right_shoulder_x = self.vec_between(right_shoulder_x, left_shoulder_x)
            left_to_right_shoulder_y = self.vec_between(right_shoulder_y, left_shoulder_y)
            msg.left_shoulder_angle = self.angle_between([left_elbow_vec_x, left_elbow_vec_y],
                                                [left_to_right_shoulder_x, left_to_right_shoulder_y])
            left_wrist_vec_x = self.vec_between(left_wrist_x, left_elbow_x)
            left_wrist_vec_y = self.vec_between(left_wrist_y, left_elbow_y)
            msg.left_elbow_angle = self.angle_between([left_wrist_vec_x, left_wrist_vec_y],
                                             [-1 * left_elbow_vec_x, -1 * left_elbow_vec_y])
            # Want distance for shoulder -> wrist. also want angle
            left_arm_vec_x = self.vec_between(left_wrist_x, left_shoulder_x)
            left_arm_vec_y = self.vec_between(left_wrist_y, left_shoulder_y)
            msg.left_wrist_angle = self.angle_between([left_arm_vec_x, left_arm_vec_y],
                                                      [left_to_right_shoulder_x, left_to_right_shoulder_y])
            msg.left_arm_vec_x = left_arm_vec_x
            msg.left_arm_vec_y = left_arm_vec_y

            right_to_left_shoulder_vec_x = self.vec_between(left_shoulder_x, right_shoulder_x)
            right_to_left_shoulder_vec_y = self.vec_between(left_shoulder_y, right_shoulder_y)
            msg.shoulder_vec_x = right_to_left_shoulder_vec_x
            msg.shoulder_vec_y = right_to_left_shoulder_vec_y

            right_elbow_vec_x = self.vec_between(right_elbow_x, right_shoulder_x)
            right_elbow_vec_y = self.vec_between(right_elbow_y, right_shoulder_y)
            msg.right_shoulder_angle = self.angle_between([right_elbow_vec_x, right_elbow_vec_y],
                                                 [right_to_left_shoulder_vec_x, right_to_left_shoulder_vec_y])
            right_wrist_vec_x = self.vec_between(right_wrist_x, right_elbow_x)
            right_wrist_vec_y = self.vec_between(right_wrist_y, right_elbow_y)
            msg.right_elbow_angle = self.angle_between([right_wrist_vec_x, right_wrist_vec_y],
                                              [-1 * right_elbow_vec_x, -1 * right_elbow_vec_y])
            # Want distance & angle for shoulder -> wrist
            right_arm_vec_x = self.vec_between(right_wrist_x, right_shoulder_x)
            right_arm_vec_y = self.vec_between(right_wrist_y, right_shoulder_y)
            msg.right_wrist_angle = self.angle_between([right_arm_vec_x, right_arm_vec_y],
                                                       [right_to_left_shoulder_vec_x, right_to_left_shoulder_vec_y])
            msg.right_arm_vec_x = right_arm_vec_x
            msg.right_arm_vec_y = right_arm_vec_y

            msg.neck_x = neck_x
            msg.neck_y = neck_y
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
