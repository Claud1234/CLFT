#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Script to select 20% lines out of a text file
'''
import os
import random


with open('/home/claude/Data/claude_iseauto/day_fair/list_2000_shuf.txt') as f:
	all_lines = list(f)


day_fair_labelled_txt = open('/home/claude/Data/claude_iseauto/day_fair/day_fair_labelled.txt', 'w')
day_fair_unlabelled_txt = open('/home/claude/Data/claude_iseauto/day_fair/day_fair_unlabelled.txt', 'w')

selected = random.sample(all_lines, int(len(all_lines)*.2))

for ele in selected:
	day_fair_labelled_txt.write(ele)

for a in all_lines:
	if a not in selected:
		day_fair_unlabelled_txt.write(a)

day_fair_labelled_txt.close()
day_fair_unlabelled_txt.close()
