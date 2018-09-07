# RAIL (Human) Pose Estimation Metapackage

[![Build Status](https://travis-ci.org/GT-RAIL/rail_pose_estimation.svg?branch=develop)](https://travis-ci.org/GT-RAIL/rail_pose_estimation)

This repository contains nodes and packages that we use for human pose estimation. All methods of pose estimation publish poses in the message formats defined in [`rail_pose_estimation_msgs`](rail_pose_estimation_msgs/).

**Note on Licensing**

We try to be as permissive with our LICENSE as we can (we use an MIT license), but sometimes our dependencies use a different license. As a whole, this repository is released under the MIT license, but the estimator package at this time is released under a non-commercial license, as mandated by the detection package that it employs.

## Contents

1. [`rail_pose_estimation_msgs`](rail_pose_estimation_msgs/) - the message definitions for packages within this directory.
1. [`rail_pose_estimator`](rail_pose_estimator/) - The primary package that we use to perform human pose estimation from RGB data.
