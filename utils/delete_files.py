#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import glob
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=False, default='config.json', help='The path of the config file')
args = parser.parse_args()
config_file = args.config

with open(config_file, 'r') as f:
    config = json.load(f)

files = glob.glob(config['Log']['logdir']+'progress_save/*.pth')

if len(files) > 1:
    latest_file = max(files, key=os.path.getctime)
    print(f'Latest file: {latest_file}')
    for file in files:
        if file != latest_file:
            os.remove(file)
            print(f'Removed: {file}')
