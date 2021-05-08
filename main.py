#!/usr/bin/env python3

import cv2
import numpy as np
import superpoint as sp
import scipy.optimize

from ouster import client, pcap
from contextlib import closing
from more_itertools import nth


def match(desc1, desc2, thresh=0.6):
    dmat = np.dot(desc1.T, desc2)
    dmat = np.sqrt(2 - 2 * np.clip(dmat, -1, 1))
    matches = scipy.optimize.linear_sum_assignment(dmat, maximize=False)
    matches = [z for z in zip(*matches) if dmat[z[0], z[1]] < thresh]
    return matches


def plotmatch(img1, img2, pts1, pts2, matches):
    img = np.concatenate((img1, img2), axis=0)
    for m1, m2 in matches:
        p1 = (int(pts1[0, m1]), int(pts1[1, m1]))
        p2 = (int(pts2[0, m2]), int(pts2[1, m2] + 128))
        cv2.line(img, p1, p2, [0, 100, 255], thickness=2)
    return img

def low_latency_odometry(xyz1, xyz2, pts1, pts2, matches):
    for m1, m2 in matches:
        p1 = xyz1[int(pts1[0, m1]), int(pts1[1, m1])]
        p2 = xyz2[int(pts2[0, m2]), int(pts2[1, m2])]


pcap_path = 'OS2_128.pcap'
metadata_path = 'OS2_1024x20_128.json'

metadata = client.SensorInfo(open(metadata_path).read())
xyzlut = client.XYZLut(metadata)

source = pcap.Pcap(pcap_path, metadata)

fe = sp.SuperPointFrontend(weights_path='superpoint_v1.pth',
                           nms_dist=4,
                           conf_thresh=0.01,
                           nn_thresh=0.6,
                           cuda=True)
print('==> Successfully loaded pre-trained network.')
frame = 0

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

zoo_desc = None
zoo_pts = None

orb = cv2.ORB_create()

prev_img = None
with closing(client.Scans(source)) as stream:
    for scan in stream:
        if frame == 0:
            frame += 1
            continue
        range_field = scan.field(client.ChanField.RANGE)
        range_img = client.destagger(source.metadata, range_field)
        intensity_field = scan.field(client.ChanField.SIGNAL)
        intensity_img = client.destagger(source.metadata, intensity_field)
        xyz_destaggered = client.destagger(metadata, xyzlut(scan))
        print('xyz', xyz_destaggered.shape)

        signal = client.destagger(stream.metadata,
                                  scan.field(client.ChanField.SIGNAL))
        signal = np.sqrt(signal)
        signal = signal / np.max(signal)
        signal = signal.astype(np.float32)
        img = (signal * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        pts, desc, heatmap = fe.run(signal)
        print('sp', pts.shape, desc.shape, type(desc.dtype))

        #pts, desc = orb.detectAndCompute(img, None)
        #pts = np.array([[k.pt[0], k.pt[1]] for k in pts])
        #pts = pts.T
        #print('orb', pts.shape, desc.shape, type(desc.dtype))
        if frame == 1:
            zoo_desc = desc
            zoo_pts = pts

            frame += 1
            prev_img = img
            continue

        matches = match(desc, zoo_desc)
        mimg = plotmatch(img, prev_img, pts, zoo_pts, matches)
        cv2.imwrite('output_{:06}.png'.format(frame), mimg)

        zoo_desc = desc
        zoo_pts = pts
        frame += 1
        prev_img = img

        cv2.imshow("img", mimg)
        key = cv2.waitKey(1) & 0xFF
