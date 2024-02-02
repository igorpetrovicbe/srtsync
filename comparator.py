import datetime
import math
import os
import string
import time

import srt
import numpy as np
from sub_fixer_medium import get_fixed_speed_offset, remove_non_ascii
import random

import json
import requests


def get_files_in_folder(folder_path, suffix, input_prefix):
    sub_list = []
    input_sub_list = []
    vid_list = []
    txt_list = []
    sub_extension = '.srt'

    # Iterate through all the items in the folder
    appended_idx = -1
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)

        # If it's a file and has the desired file extension, add it to the list
        if os.path.isfile(item_path):
            if appended_idx == -1:
                txt_list.append(False)
                appended_idx = len(txt_list) - 1
            if item.endswith(sub_extension):
                if item.endswith(suffix + sub_extension):
                    sub_list.append(item_path)
                elif item.startswith(input_prefix):
                    input_sub_list.append(item_path)
            elif not item.endswith('.txt'):
                vid_list.append(item_path)
            else:
                txt_list[appended_idx] = True

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)

        # If it's a directory, recursively call the function on that directory
        if os.path.isdir(item_path):
            new_subs, new_vids, new_txts, new_input_subs = get_files_in_folder(item_path, suffix, input_prefix)
            sub_list.extend(new_subs)
            vid_list.extend(new_vids)
            txt_list.extend(new_txts)
            input_sub_list.extend(new_input_subs)

    return sub_list, vid_list, txt_list, input_sub_list


def import_all_subs(folder_path):
    subs = []
    sub_names = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)

        if os.path.isfile(item_path):
            subs.append(item_path)
            sub_names.append(item)

    return subs, sub_names


if __name__ == '__main__':

    current_dir = os.getcwd()
    print(os.getcwd())

    relative_truth_folder_path = 'output_subs/odlicni test 1'
    relative_check_folder_path = 'output_subs/results subsync-transcript/test 2 new'

    truth_folder_path = os.path.join(current_dir, relative_truth_folder_path)
    check_folder_path = os.path.join(current_dir, relative_check_folder_path)

    truth_sub_path_list, truth_sub_names = import_all_subs(truth_folder_path)
    check_sub_path_list, check_sub_names = import_all_subs(check_folder_path)

    print(str(len(truth_sub_path_list)) + ', ' + str(len(check_sub_path_list)))

    for i in range(len(truth_sub_path_list)):
        truth_sub_path = truth_sub_path_list[i]
        check_sub_path = check_sub_path_list[i]

        print(f"truth: {truth_sub_path[-60:]} | check: {check_sub_path[-60:]}")

    check_subs = []
    truth_subs = []
    start_from = 0

    test_idx = 2

    if test_idx != 3:
        do_check_list = []
    else:
        do_check_list = [0,6,7,9,13,15,16,17,18,19,20,21,22,23,24,25,26,28,32,33,34,35,36,39,40,45,46,47,51,52,55,56,57,60]

    counter = 0
    for i in range(start_from, len(truth_sub_path_list)):
        truth_sub_path = truth_sub_path_list[i]
        check_sub_path = check_sub_path_list[i]

        truth_sub_name = truth_sub_names[i]
        num_str = ''
        for j in range(len(truth_sub_name)):
            if truth_sub_name[j].isnumeric():
                num_str += truth_sub_name[j]
            else:
                break

        sub_index = int(num_str)

        if len(do_check_list) > 0 and sub_index not in do_check_list:
            truth_subs.append(0)
            check_subs.append(0)
            continue

        truth_data = None
        try:
            f = open(truth_sub_path, "r", errors="ignore")
            truth_data = f.read()
            f.close()
        except:
            out_file = open("Output.txt", "a")
            out_file.write(str(i) + ": Failed - can't open truth " + str(i) + '\n')
            out_file.close()
            truth_subs.append(0)
            check_subs.append(0)
            continue

        check_data = None
        try:
            f = open(check_sub_path, "r", errors="ignore")
            check_data = f.read()
            f.close()
        except:
            out_file = open("Output.txt", "a")
            out_file.write(str(i) + ": Failed - can't open check " + str(i) + '\n')
            out_file.close()
            truth_subs.append(0)
            check_subs.append(0)
            continue

        truth_data = remove_non_ascii(truth_data)
        check_data = remove_non_ascii(check_data)

        truth_subs.append(list(srt.parse(truth_data)))
        check_subs.append(list(srt.parse(check_data)))

        truth_sub = truth_subs[i]
        check_sub = check_subs[i]

        dif = len(truth_sub) - len(check_sub)
        print(str(i) + ') ' + str(sub_index) + ', ' + str(dif))

        sample_size = math.ceil(len(check_sub) * 0.1)

        sample_start_total = 0
        sample_end_total = 0

        first_line = 0
        match_count = 0

        if test_idx == 1 or test_idx == 3:
            for j in range(len(truth_sub)):
                if truth_sub[j].content == check_sub[match_count].content:  # Ovde sam zamenio gde je j
                    if match_count == 0:
                        first_line = j
                    match_count += 1
                else:
                    match_count = 0
                if match_count == 3:
                    break

            for j in range(sample_size):
                sample_start_total += check_sub[j].start.total_seconds() - truth_sub[j + first_line].start.total_seconds()
                sample_end_total += check_sub[-j-1].start.total_seconds() - truth_sub[-j-1].start.total_seconds()
        elif test_idx == 2:
            for j in range(len(check_sub)):
                if truth_sub[match_count].content == check_sub[j].content:
                    if match_count == 0:
                        first_line = j
                    match_count += 1
                else:
                    match_count = 0
                if match_count == 3:
                    break

            print('First line: ' + str(first_line))
            if first_line > 100:
                print("skipped " + str(sub_index))
                continue
            for j in range(sample_size):
                sample_start_total += check_sub[j+first_line].start.total_seconds() - truth_sub[j].start.total_seconds()
                sample_end_total += check_sub[-j-1].start.total_seconds() - truth_sub[-j-1].start.total_seconds()

        start_error = sample_start_total / sample_size
        end_error = sample_end_total / sample_size

        out_file = open("Output comparison.txt", "a")
        out_file.write(str(i) + ') ' + str(sub_index) + '_' + truth_sub_path_list[i][-14:] + ' ' + str(start_error) + ', ' +
                       str(end_error) + ' ' + '\n')
        out_file.close()
