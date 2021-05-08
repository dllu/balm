#!/usr/bin/env python3

import cv2
import numpy as np
import superpoint as sp
import scipy.optimize
from scipy.spatial.transform import Rotation

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


def skewsym(a):
    return np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]],
                     [-a[1], a[0], 0.0]])


def se3exp(xi):
    out = np.identity(4)
    out[:3, :3] = Rotation.from_rotvec(xi[:3].T).as_matrix()
    out[:3, 3] = xi[3:].T
    return out


def low_latency_odometry(xyz1, xyz2, pts1, pts2, ts1, ts2, matches, ts0):
    jacobian = np.zeros((len(matches) * 3, 6))
    residual = np.zeros((len(matches) * 3, 1))

    odom = np.zeros((6, 1))

    for i, mm in enumerate(matches):
        m1, m2 = mm
        x1, y1 = int(pts1[0, m1]), int(pts1[1, m1])
        x2, y2 = int(pts2[0, m2]), int(pts2[1, m2])

        t1 = (int(ts1[y1, x1]) - int(ts0)) / 1e9
        t2 = (int(ts2[y2, x2]) - int(ts0)) / 1e9

        p1 = xyz1[y1, x1]
        p2 = xyz2[y2, x2]

        homo1 = se3exp(odom * t1)
        homo2 = se3exp(odom * t2)
        moved_p1 = homo1[:3, :3] @ p1 + homo1[:3, 3]
        moved_p2 = homo2[:3, :3] @ p2 + homo2[:3, 3]

        block = np.zeros((3, 6))
        block[:, :3] = t1 * -skewsym(moved_p1) - t2 * -skewsym(moved_p2)
        block[:, 3:] = (t1 - t2) * np.identity(3)
        jacobian[(3 * i):(3 * i + 3), :] = block
        residual[(3 * i):(3 * i + 3), :] = np.array([moved_p1 - moved_p2]).T
    update = np.linalg.solve(jacobian.T @ jacobian, -jacobian.T @ residual)
    print(update)


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

zoo_desc = None
zoo_pts = None

orb = cv2.ORB_create()

prev_img = None
prev_ts = None
with closing(client.Scans(source)) as stream:
    for scan in stream:
        if frame == 0:  # first frame usually incomplete garbage
            frame += 1
            continue
        range_field = scan.field(client.ChanField.RANGE)
        range_img = client.destagger(source.metadata, range_field)
        intensity_field = scan.field(client.ChanField.SIGNAL)
        intensity_img = client.destagger(source.metadata, intensity_field)

        timestamps = scan.header(client.ColHeader.TIMESTAMP)
        ts0 = timestamps[0]
        print(type(timestamps.dtype))
        timestamps = np.tile(timestamps, (128, 1))  # LOL
        ts = client.destagger(metadata, timestamps)

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
            prev_ts = ts
            prev_xyz = xyz_destaggered
            continue

        matches = match(desc, zoo_desc)
        mimg = plotmatch(img, prev_img, pts, zoo_pts, matches)
        odom = low_latency_odometry(xyz_destaggered, prev_xyz, pts, zoo_pts,
                                    ts, prev_ts, matches, ts0)
        cv2.imwrite('output_{:06}.png'.format(frame), mimg)

        zoo_desc = desc
        zoo_pts = pts
        frame += 1
        prev_img = img
        prev_ts = ts
        prev_xyz = xyz_destaggered

        cv2.imshow("img", mimg)
        key = cv2.waitKey(1) & 0xFF
