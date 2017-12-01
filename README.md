# rail_cpm

Notably lacking the models directory (and requiring an absolute path to that missing directory). Contact me to get the model, since it's too big to push up with everything else.
This also expects that caffe is in your $PYTHONPATH, so if it is not then ensure you do the sys.path workaround.

### What is published?
This node will publish /rail_cpm/poses, which is a list of people found in the scene. Each pose/person object will have 2 points for every keypoint, the Y and X coordinate. So the pose object will contain "nose_y, nose_x, neck_y, neck_x, right_shoulder_y, right_shoulder_x" and so on. If the keypoint is not found for that person, it is returned as -1.0. It is important to note that (Y, X) corresponds to (Column, Row) in the image. So Y is the number of pixels from the left border, X is the number from the top border. For a visualization, run with the debug flag and run:
```
rosrun image_view image_view image:=/rail_cpm/debug/keypoint_image
```

### Running:
To actually run the node, launch with:
```
roslaunch rail_cpm detector.launch
```
optional flags include 
  * 'debug' (true or false)
  * 'image_sub_topic_name' which is /kinect/qhd/image_color_rect by default.


### Editing things:
If you want to change the keypoints that are returned, you need to mess around with lines 218-237. I have hard-coded the limb-sequences that I am interested in, which means that knees, hips, and ankles aren't coming through. Currently, the code is going
```
for i in range(6):
  ...
for i in range(12, 19):
  ...
```
If you want to get all keypoints, all you need to do is cut out one for loop and extend the range of the other, for example:
```
for i in range(19):
  ...
```
And actually, 18 might work too...

### Warnings:
These only really matter for human interpretability, but they should be noted before I forget. First, as mentioned above: Y and X are flipped from how most people would interpret them. X is row (down) and Y is column (across). Second, points will gravitate to borders if somebody is too close. So instead of being 5px away from the edge, they'll just snap over. It looks like a division error somewhere, I'll try to fix it at some point. But for now just be aware of it!

For other questions, just reach out!
# rail_pose_inprogress
