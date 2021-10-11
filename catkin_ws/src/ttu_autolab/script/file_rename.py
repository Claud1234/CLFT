#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Script to add the squence number before the file name'
'''
import os


path_0 = '/media/usb/TalTech_DriverlessProject/catkin_ws/src/ttu_autolab/output/day_fair/sq12/bin'
path_1 = '/media/usb/TalTech_DriverlessProject/catkin_ws/src/ttu_autolab/output/day_fair/sq12/pcd'
path_2 = '/media/usb/TalTech_DriverlessProject/catkin_ws/src/ttu_autolab/output/day_fair/sq12/bin'
path_3 = '/media/usb/TalTech_DriverlessProject/catkin_ws/src/ttu_autolab/output/day_fair/sq12/bin'
path_4 = '/media/usb/TalTech_DriverlessProject/catkin_ws/src/ttu_autolab/output/day_fair/sq12/bin'

string = 'sq12_'

files_0 = os.listdir(path_0)
files_1 = os.listdir(path_1)
files_2 = os.listdir(path_2)
files_3 = os.listdir(path_3)
files_4 = os.listdir(path_4)


for i, name in enumerate(files_0):
#     os.rename(os.path.join(path_0, name), os.path.join(path_0, name.split('_')[1]))
    os.rename(os.path.join(path_0, name), os.path.join(path_0, string + name))
 
for i, name in enumerate(files_1):
    os.rename(os.path.join(path_1, name), os.path.join(path_1, string + name))

for i, name in enumerate(files_2):
    os.rename(os.path.join(path_2, name), os.path.join(path_2, string + name))

for i, name in enumerate(files_3):
    os.rename(os.path.join(path_3, name), os.path.join(path_3, string + name))

for i, name in enumerate(files_4):
    os.rename(os.path.join(path_4, name), os.path.join(path_4, string + name))
