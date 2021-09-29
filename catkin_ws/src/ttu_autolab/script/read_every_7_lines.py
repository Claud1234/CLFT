#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Script to read every 7 lines in a txt file.
'''

with open('list.txt', 'r') as f:
    with open('list1.txt', 'w') as g:
        count = 0
        for line in f:
            if count % 7 == 0:
                g.write(line)
            count += 1
