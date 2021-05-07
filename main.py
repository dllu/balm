#!/usr/bin/env python3

import cv2
import numpy as np
import superpoint as sp

from ouster import client, pcap
from contextlib import closing
from more_itertools import nth

pcap_path = 'OS2_128.pcap'
metadata_path = 'OS2_1024x20_128.json'

with open(metadata_path, 'r') as f:
    metadata = client.SensorInfo(f.read())

source = pcap.Pcap(pcap_path, metadata)

fe = sp.SuperPointFrontend(weights_path='superpoint_v1.pth',
                           nms_dist=4,
                           conf_thresh=0.01,
                           nn_thresh=0.6,
                           cuda=True)
print('==> Successfully loaded pre-trained network.')
point_tracker = sp.PointTracker(max_length=5, nn_thresh=fe.nn_thresh)

with closing(client.Scans(source)) as stream:
    for scan in stream:
        range_field = scan.field(client.ChanField.RANGE)
        range_img = client.destagger(source.metadata, range_field)
        intensity_field = scan.field(client.ChanField.SIGNAL)
        intensity_img = client.destagger(source.metadata, intensity_field)

        signal = client.destagger(stream.metadata,
                                  scan.field(client.ChanField.SIGNAL))
        signal = np.sqrt(signal)
        signal = signal / np.max(signal)
        signal = signal.astype(np.float32)

        pts, desc, heatmap = fe.run(signal)
        point_tracker.update(pts, desc)
        point_tracks = point_tracker.get_tracks(2)

        img = (signal * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        point_tracker.draw_tracks(img, point_tracks)

        cv2.imshow("img", img)
        key = cv2.waitKey(1) & 0xFF
