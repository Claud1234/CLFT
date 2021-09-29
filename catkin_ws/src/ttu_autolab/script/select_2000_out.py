#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Script to read list_200.txt and copy the files written in it.
'''
import os
import shutil

with open('/media/usb/TalTech_DriverlessProject/catkin_ws/src/ttu_autolab/output/day_fair/list_2000.txt') as f:
    lines = [line.rstrip() for line in f]

for root, dirs, files in os.walk('/media/usb/TalTech_DriverlessProject/catkin_ws/src/ttu_autolab/output/day_fair/lidar_rgb'):
    for i, name in enumerate(lines):
        if lines[i] in files:
            shutil.copy2(os.path.join(root, lines[i]), '/home/claude/Data/claude_iseauto/lidar_rgb/')
