import datetime
import math
import os
import string

import srt
import numpy as np
from sub_fixer_medium import get_fixed_speed_offset, remove_non_ascii
import random

import copy
import json
import requests


def get_files_in_folder(folder_path, suffix):
    sub_list = []
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
            elif not item.endswith('.txt'):
                vid_list.append(item_path)
            else:
                txt_list[appended_idx] = True

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)

        # If it's a directory, recursively call the function on that directory
        if os.path.isdir(item_path):
            new_subs, new_vids, new_txts = get_files_in_folder(item_path, suffix)
            sub_list.extend(new_subs)
            vid_list.extend(new_vids)
            txt_list.extend(new_txts)

    return sub_list, vid_list, txt_list


if __name__ == '__main__':

    current_dir = os.getcwd()
    print(os.getcwd())

    relative_root_folder_path = 'dataset/test'
    suffix = "correct"
    root_folder_path = os.path.join(current_dir, relative_root_folder_path)

    # Call the function to get all files with the specified extension
    sub_path_list, vid_path_list, txt_list = get_files_in_folder(root_folder_path, suffix)

    vid_file_names = []

    for i in range(len(sub_path_list)):
        sub_path = sub_path_list[i]
        vid_path = vid_path_list[i]

        last_occurrence_index = vid_path.rfind('\\')
        vid_file_name = sub_path[last_occurrence_index + 1:]
        vid_file_names.append(vid_file_name)

        print(f"sub: {sub_path[-50:]} | vid: {vid_file_name}")

    input_subs = []
    sub_offsets = []
    correct_subs = []
    start_from = 0
    do_only_list = []
    do_transcription_list = []
    for i in range(len(txt_list)):
        if txt_list[i] is False:
            do_transcription_list.append(i)
            print("Doing transcription for: " + str(i))

    for i in range(start_from):
        input_subs.append(0)
        correct_subs.append(0)
        sub_offsets.append(0)

    total_start_error = 0
    total_end_error = 0
    counter = 0
    for i in range(start_from, len(sub_path_list)):
        sub_path = sub_path_list[i]

        if len(do_only_list) > 0 and i not in do_only_list:
            input_subs.append(0)
            correct_subs.append(0)
            sub_offsets.append(0)
            continue

        data = None
        try:
            f = open(sub_path, "r", errors="ignore")
            data = f.read()
            f.close()
        except:
            out_file = open("Output.txt", "a")
            out_file.write(str(i) + ": Failed - can't open " + vid_file_names[i] + '\n')
            out_file.close()
            input_subs.append(0)
            correct_subs.append(0)
            sub_offsets.append(0)
            continue

        data = remove_non_ascii(data)
        sub = list(srt.parse(data))
        correct_subs.append(list(srt.parse(data)))

        offset = random.uniform(60, 180)
        speed = random.uniform(0.9, 1.1)

        offset_time = datetime.timedelta(seconds=offset)

        correct_sub = correct_subs[i]
        print(sub[0], " ; ", correct_sub[0])
        print(sub[-1], " ; ", correct_sub[-1])
        print(i, ": ", abs(sub[0].start.total_seconds() - correct_sub[0].start.total_seconds()), " , ",
              abs(sub[-1].start.total_seconds() - correct_sub[-1].start.total_seconds()))

        for line in sub:
            line.start = speed * line.start + offset_time
            line.end = speed * line.end + offset_time

        input_subs.append(sub)
        sub_offsets.append(offset)

    # Add dialogue lines
    for i in range(len(input_subs)):
        if input_subs[i] == 0:
            continue
        intro_sub_idx = i
        while intro_sub_idx == i or input_subs[intro_sub_idx] == 0:
            intro_sub_idx = random.randint(0, len(input_subs) - 1)
        intro_sub = input_subs[intro_sub_idx]
        intro_offset = sub_offsets[i]
        start_percent = random.uniform(0.3, 0.7)
        start_idx = math.floor(len(intro_sub) * start_percent)

        num_remaining_lines = len(intro_sub) - start_idx - 1
        line_num_multiplier = intro_offset / 60

        num_lines = min(num_remaining_lines, math.floor(random.randint(10, 20) * line_num_multiplier))

        intro_lines = []
        for j in range(start_idx, start_idx + num_lines):
            intro_lines.append(copy.deepcopy(intro_sub[j]))

        min_time = intro_lines[0].start.total_seconds()
        max_time = intro_lines[-1].end.total_seconds()
        scaling_factor = intro_offset / (max_time - min_time)

        for line in intro_lines:
            line.start = scaling_factor * (line.start - datetime.timedelta(seconds=min_time))
            line.end = scaling_factor * (line.end - datetime.timedelta(seconds=min_time))

        input_subs[i] = intro_lines + input_subs[i]

    # Export input sub
    for i in range(len(input_subs)):
        sub = input_subs[i]
        input_sub_out_path = os.path.join(current_dir, 'used_inputs/inputs test 2/k2_' + str(i) + ".srt")
        input_sub_out_file = open(input_sub_out_path, "w")
        input_sub_out_file.write(srt.compose(sub))
        input_sub_out_file.close()

