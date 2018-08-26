# !/usr/bin/env python

"""
Pose Detector Object for ROS
"""

import cv2 as cv
import numpy as np
import time
import math
import rospkg
from scipy.ndimage.filters import gaussian_filter
import os
# import sys
# caffe_root = '/home/asilva/caffe/python'
# sys.path.append(caffe_root)
import caffe

### Begin Config:

param = {'use_gpu': 1, 'GPUdeviceNumber': 0, 'modelID': '1', 'octave': 3, 'starting_range': 0.8, 'ending_range': 2.0,
         'scale_search': [0.5, 1.0, 1.5, 2.0], 'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5, 'min_num': 4, 'mid_num': 10,
         'crop_ratio': 2.5, 'bbox_ratio': 0.25}
rospack = rospkg.RosPack()
parent_dir = rospack.get_path('rail_pose_estimator')
model_dir = 'model/'
model_file = 'pose_iter_440000.caffemodel'
model_proto = 'pose_deploy.prototxt'
model_path = os.path.join(parent_dir, model_dir, model_file)
proto_path = os.path.join(parent_dir, model_dir, model_proto)
model = {'caffemodel': model_path,
         'deployFile': proto_path,
         'description': 'COCO Pose56 Two-level Linevec',
         'boxsize': 368, 'padValue': 128, 'np': '12', 'stride': 8,
         'part_str': ['nose', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'left_shoulder', 'left_elbow',
                      'left_wrist', 'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'right_eye', 'left_eye', 'right_ear',
                      'left_ear', 'pt19']}
down_scale = 0.5

limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12], [12, 13], [13, 14],
           [2, 1], [1, 15], [15, 17],[1, 16], [16, 18], [3, 17], [6, 18]]
# the middle joints heatmap correpondence
mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], [23, 24], [25, 26], [27, 28],
          [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], [55, 56], [37, 38], [45, 46]]
### End Config


class PoseMachine:
    def __init__(self):

        caffe.set_mode_gpu()
        caffe.set_device(0)
        self.net = caffe.Net(model['deployFile'], model['caffemodel'], caffe.TEST)
        self.scale = 0
        print '\n\nLoaded network {:s}'.format(model['caffemodel'])

    def padRightDownCorner(self, img, stride, padValue):
        h = img.shape[0]
        w = img.shape[1]

        pad = 4 * [None]
        pad[0] = 0  # up
        pad[1] = 0  # left
        pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
        pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

        img_padded = img
        pad_up = np.tile(img_padded[0:1, :, :] * 0 + padValue, (pad[0], 1, 1))
        img_padded = np.concatenate((pad_up, img_padded), axis=0)
        pad_left = np.tile(img_padded[:, 0:1, :] * 0 + padValue, (1, pad[1], 1))
        img_padded = np.concatenate((pad_left, img_padded), axis=1)
        pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + padValue, (pad[2], 1, 1))
        img_padded = np.concatenate((img_padded, pad_down), axis=0)
        pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + padValue, (1, pad[3], 1))
        img_padded = np.concatenate((img_padded, pad_right), axis=1)
        return img_padded, pad

    def estimate_keypoints(self, input_image):
        caffe.set_mode_gpu()
        caffe.set_device(0)
        start_time = time.time()
        self.scale = down_scale * model['boxsize'] / input_image.shape[0]
        imageToTest = cv.resize(input_image, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv.INTER_CUBIC)
        imageToTest_padded, pad = self.padRightDownCorner(imageToTest, model['stride'], model['padValue'])
        self.net.blobs['data'].reshape(*(1, 3, imageToTest_padded.shape[0], imageToTest_padded.shape[1]))
        self.net.blobs['data'].data[...] = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5;
        caffe.set_mode_gpu()
        caffe.set_device(0)
        output_blobs = self.net.forward()
        print 'CNN time: ', time.time()-start_time

        # Visualize detections for each class
        heatmap = np.transpose(np.squeeze(self.net.blobs[output_blobs.keys()[1]].data), (1, 2, 0))  # output 1 is heatmaps
        heatmap = cv.resize(heatmap, (0, 0), fx=model['stride'], fy=model['stride'], interpolation=cv.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = np.transpose(np.squeeze(self.net.blobs[output_blobs.keys()[0]].data), (1, 2, 0))  # output 0 is PAFs
        paf = cv.resize(paf, (0,0), fx=model['stride'], fy=model['stride'], interpolation=cv.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
        peak_counter = 0
        all_peaks = []
        for part in range(18):
            map_ori = heatmap[:, :, part]
            map = gaussian_filter(map_ori, sigma=3)

            map_left = np.zeros(map.shape)
            map_left[1:, :] = map[:-1, :]
            map_right = np.zeros(map.shape)
            map_right[:-1, :] = map[1:, :]
            map_up = np.zeros(map.shape)
            map_up[:, 1:] = map[:, :-1]
            map_down = np.zeros(map.shape)
            map_down[:, :-1] = map[:, 1:]

            peaks_binary = np.logical_and.reduce(
                (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > param['thre1']))
            peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
            id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]
            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)
        # print 'Total time for estimation: ', time.time() - start_time
        connection_all = []
        special_k = []
        mid_num = 10
        for k in range(len(mapIdx)):
            score_mid = paf[:, :, [x - 19 for x in mapIdx[k]]]
            candA = all_peaks[limbSeq[k][0] - 1]
            candB = all_peaks[limbSeq[k][1] - 1]
            nA = len(candA)
            nB = len(candB)
            indexA, indexB = limbSeq[k]
            if (nA != 0 and nB != 0):
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        vec = np.subtract(candB[j][:2], candA[i][:2])
                        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                        # Silva divide by zero fix?
                        if norm == 0:
                            continue
                        vec = np.divide(vec, norm)
                        startend = zip(np.linspace(candA[i][0], candB[j][0], num=mid_num),
                                       np.linspace(candA[i][1], candB[j][1], num=mid_num))

                        vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0]
                                          for I in range(len(startend))])
                        vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1]
                                          for I in range(len(startend))])

                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        # Silva divide by zero fix
                        if norm == 0:
                            score_with_dist_prior = sum(score_midpts) / len(score_midpts)
                        else:
                            score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                                0.5 * input_image.shape[0] / norm - 1, 0)
                        criterion1 = len(np.nonzero(score_midpts > param['thre2'])[0]) > 0.8 * len(score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append(
                                [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if (i not in connection[:, 3] and j not in connection[:, 4]):
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                        if (len(connection) >= min(nA, nB)):
                            break

                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])
        # last number in each row is the total parts number of that person
        # the second last number in each row is the score of the overall configuration
        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in all_peaks for item in sublist])

        for k in range(len(mapIdx)):
            if k not in special_k:
                partAs = connection_all[k][:, 0]
                partBs = connection_all[k][:, 1]
                indexA, indexB = np.array(limbSeq[k]) - 1

                for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):  # 1:size(subset,1):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if (subset[j][indexB] != partBs[i]):
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2:  # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        print "found = 2"
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:  # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])
        # delete some rows of subset which has few parts occur
        deleteIdx = [];
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                deleteIdx.append(i)
        subset = np.delete(subset, deleteIdx, axis=0)
        # Silva's temp hack. return early if nobody shows up. don't know why this wasn't here already?
        if len(candidate) == 0:
            return []
        candidate[:, 0] *= 1.0 / self.scale
        candidate[:, 1] *= 1.0 / self.scale
        people = []
        for n in (range(len(subset))):
            people.append({})
            for i in range(len(limbSeq)):
                index = subset[n][np.array(limbSeq[i]) - 1]
                if -1 in index:
                    continue
                y_arr = candidate[index.astype(int), 0]
                x_arr = candidate[index.astype(int), 1]
                part_in = np.array(limbSeq[i]) - 1
                # print model['part_str'][part_in[0]]
                people[n][model['part_str'][part_in[0]]] = (y_arr[0], x_arr[0])
                people[n][model['part_str'][part_in[1]]] = (y_arr[1], x_arr[1])
        # print 'Total time for full: ', time.time() - start_time
        return people

    def visualize_keypoints(self, canvas, people):
        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
                      [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
                      [85, 0, 255],
                      [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
        for person in people:
            i = 0
            for key, val in person.iteritems():
                cv.circle(canvas, (int(val[0]), int(val[1])), 5, colors[i], thickness=-1)
                i += 1
        return canvas
