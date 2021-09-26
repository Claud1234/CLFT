#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random


day_fair_parent_path = 'labeled/day/not_rain/camera/'
day_rain_parent_path = 'labeled/day/rain/camera/'
night_fair_parent_path = 'labeled/night/not_rain/camera/'
night_rain_parent_path = 'labeled/night/rain/camera/'

day_fair_files = os.listdir(day_fair_parent_path)
day_rain_files = os.listdir(day_rain_parent_path)
night_fair_files = os.listdir(night_fair_parent_path)
night_rain_files = os.listdir(night_rain_parent_path)

day_fair_eval = random.sample(day_fair_files, int(len(day_fair_files)*.1))
day_rain_eval = random.sample(day_rain_files, int(len(day_rain_files)*.3))
night_fair_eval = random.sample(night_fair_files, int(len(night_fair_files)*.3))
night_rain_eval = random.sample(night_rain_files, int(len(night_rain_files)*.3))

other_day_fair = [a for a in day_fair_files if a not in day_fair_eval]
other_day_rain = [b for b in day_rain_files if b not in day_rain_eval]
other_night_fair = [c for c in night_fair_files if c not in night_fair_eval]
other_night_rain = [d for d in night_rain_files if d not in night_rain_eval]

print(len(other_day_rain))

day_fair_valid = random.sample(other_day_fair, int(len(day_fair_files)*.1))
day_rain_valid = random.sample(other_day_rain, int(len(day_rain_files)*.1))
night_fair_valid = random.sample(other_night_fair, int(len(night_fair_files)*.1))
night_rain_valid = random.sample(other_night_rain, int(len(night_rain_files)*.1))

print(len(day_rain_valid))

rest_day_fair = [x for x in other_day_fair if x not in day_fair_valid]
rest_day_rain = [y for y in other_day_rain if y not in day_rain_valid]
rest_night_fair = [z for z in other_night_fair if z not in night_fair_valid]
rest_night_rain = [t for t in other_night_rain if t not in night_rain_valid]
print(len(rest_day_fair))

day_fair_eval_txt = open('eval_day_fair.txt', 'w')
day_rain_eval_txt = open('eval_day_rain.txt', 'w')
night_fair_eval_txt = open('eval_night_fair.txt', 'w')
night_rain_eval_txt = open('eval_night_rain.txt', 'w')

early_stop_valid_txt = open('early_stop_valid.txt', 'w')
all_train_txt = open('train_all.txt', 'w')

for elem in day_fair_eval:
    day_fair_eval_txt.write(day_fair_parent_path + elem + '\n')
day_fair_eval_txt.close()
for elem in day_rain_eval:
    day_rain_eval_txt.write(day_rain_parent_path + elem + '\n')
day_rain_eval_txt.close()
for elem in night_fair_eval:
    night_fair_eval_txt.write(night_fair_parent_path + elem + '\n')
night_fair_eval_txt.close()
for elem in night_rain_eval:
    night_rain_eval_txt.write(night_rain_parent_path + elem + '\n')
night_rain_eval_txt.close()

for elem in day_fair_valid:
    early_stop_valid_txt.write(day_fair_parent_path + elem + '\n')
for elem in day_rain_valid:
    early_stop_valid_txt.write(day_rain_parent_path + elem + '\n')
for elem in night_fair_valid:
    early_stop_valid_txt.write(night_fair_parent_path + elem + '\n')
for elem in night_rain_valid:
    early_stop_valid_txt.write(night_rain_parent_path + elem + '\n')
early_stop_valid_txt.close()

for elem in rest_day_fair:
    all_train_txt.write(day_fair_parent_path + elem + '\n')
for elem in rest_day_rain:
    all_train_txt.write(day_rain_parent_path + elem + '\n')
for elem in rest_night_fair:
    all_train_txt.write(night_fair_parent_path + elem + '\n')
for elem in rest_night_rain:
    all_train_txt.write(night_rain_parent_path + elem + '\n')
all_train_txt.close()


lines_0 = open('early_stop_valid.txt').readlines()
random.shuffle(lines_0)
open('early_stop_valid.txt', 'w').writelines(lines_0)

lines_1 = open('train_all.txt').readlines()
random.shuffle(lines_1)
open('train_all.txt', 'w').writelines(lines_1)
