### Warnings:
These only really matter for human interpretability, but they should be noted before I forget. First, as mentioned above: Y and X are flipped from how most people would interpret them. X is row (down) and Y is column (across). Second, points will gravitate to borders if somebody is too close. So instead of being 5px away from the edge, they'll just snap over. It looks like a division error somewhere, I'll try to fix it at some point. But for now just be aware of it!
For other questions, just reach out!

# rail_pose_estimator
Multi-person pose estimation node using caffe and python

## Two Minute Intro

This detector uses [Caffe](http://caffe.berkeleyvision.org/) to perform pose estimation. It publishes poses for people found in images from a subscribed image topic. The pose estimator itself can be found here: https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation . 

Each pose/person object will have 2 points for every keypoint, the Y and X coordinate. So the pose object will contain "nose_y, nose_x, neck_y, neck_x, right_shoulder_y, right_shoulder_x" and so on. If the keypoint is not found for that person, it is returned as -1.0. It is important to note that (Y, X) corresponds to (Column, Row) in the image. So Y is the number of pixels from the left border, X is the number from the top border. For a visualization, run with the debug flag and run:
```
rosrun image_view image_view image:=/rail_pose_estimator_node/debug/poses_image
```

You should see something like this:
![Pose estimator visualization](poses.gif)

The message type coming back from the face detector is a Poses.msg which contains an array of Keypoints.msg objects. Each Keypoints.msg has:
```
float32 neck_x                  # x coord of neck
float32 neck_y                  # y coord of neck
float32 nose_x                  # x coord of nose
float32 nose_y                  # y coord of nose
float32 right_shoulder_x        # x coord of right shoulder
float32 right_shoulder_y        # y coord of right shoulder
float32 left_shoulder_x         # x coord of left shoulder
float32 left_shoulder_y         # y coord of left shoulder
float32 right_elbow_x           # x coord of right elbow
float32 right_elbow_y           # y coord of right elbow
float32 left_elbow_x            # x coord of left elbow
float32 left_elbow_y            # y coord of left elbow
float32 right_wrist_x           # x coord of right wrist
float32 right_wrist_y           # y coord of right wrist
float32 left_wrist_x            # x coord of left wrist
float32 left_wrist_y            # y coord of left wrist
float32 right_hip_x             # x coord of right hip
float32 right_hip_y             # y coord of right hip
float32 left_hip_x              # x coord of left hip
float32 left_hip_y              # y coord of left hip
float32 right_knee_x            # x coord of right knee
float32 right_knee_y            # y coord of right knee
float32 left_knee_x             # x coord of left knee
float32 left_knee_y             # y coord of left knee
float32 right_ankle_x           # x coord of right ankle
float32 right_ankle_y           # y coord of right ankle
float32 left_ankle_x            # x coord of left ankle
float32 left_ankle_y            # y coord of left ankle
float32 right_eye_x             # x coord of right eye
float32 right_eye_y             # y coord of right eye
float32 left_eye_x              # x coord of left eye
float32 left_eye_y              # y coord of left eye
float32 right_ear_x             # x coord of right ear
float32 right_ear_y             # y coord of right ear
float32 left_ear_x              # x coord of left ear
float32 left_ear_y              # y coord of left ear
```

## Menu
 * [Installation](#installation)
 * [Testing your Installation](#testing-your-installation)
 * [ROS Nodes](#ros-nodes)
 * [Startup](#startup)

## Installation

1. Install Caffe and PyCaffe (following instructions at http://caffe.berkeleyvision.org/installation.html)
1. (Optional) Add Caffe to your $PYTHONPATH (as instructed in the Caffe installation instructions). If you choose not to do this, you'll need to do the `sys.path.append` workaround, which I'll explain further below.
1. Download the model parameters from [this Dropbox link](https://www.dropbox.com/s/p1yohhpn40axh0r/pose_iter_440000.caffemodel?dl=0) and move them into the `model` subdirectory of this package.
1. Add this package to your ROS workspace
1. Run `catkin_make` and enjoy the ability to use pose estimation!

If caffe is not in your PYTHONPATH, you will need to explicitly point the node to your `/caffe/python` directory. To do this:
1. Open `scripts/pose_estimation.py` in this package
1. Near the top (lines 14-16) set a variable named `caffe_root` to the absolute path of your `/caffe/python` directory. There is a commented example for Ubuntu already there.
1. Uncomment the `sys.path.append(caffe_root)` line (line 16)

The node should now run properly.

## Testing your Installation

- Set up an image topic that this node can subscribe to
- Launch the `rail_pose_estimator_node` node with the debug flag and your chosen image topic name (for example, with a Kinect):
```
roslaunch rail_pose_estimator detector.launch image_sub_topic_name:=/kinect/qhd/image_color_rect debug:=true
```

## ROS Nodes

### rail_pose_estimator_node

Wrapper for object detection through ROS services.  Relevant services and parameters are as follows:

* **Topics**
  * `rail_pose_estimator_node/poses` ([rail_pose_estimator_node/poses](msg/Poses.msg))
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Topic with face detections performed in the background by running on images as they come in the subscriber.
  * `detector_node/debug/poses_image` ([rail_pose_estimator_node/debug/object_image])
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Topic with object detections visualized on incoming images as they come in from the subscriber. Only published if `debug:=true`.
* **Parameters**
  * `image_sub_topic_name` (`string`, default: "/kinect/qhd/image_color_rect")
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Image topic name to use for detections.
  * `debug` (`bool`, default: false)
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Enable or disable debug mode, which publishes incoming images with bounding boxes over faces
  * `use_compressed_image` (`bool`, default: false)
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Change to compressed image stream or not. Simply appends a "/compressed" to the image topic name. This lightens the load your local network if the images are being transmitted to the detector.

## Startup

Simply run the launch file to bring up all of the package's functionality:
```
roslaunch rail_pose_estimator detector.launch
```