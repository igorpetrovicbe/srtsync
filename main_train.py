import datetime
import math
import os
import string

import srt
import numpy as np
from sub_fixer2 import get_fixed_speed_offset, remove_non_ascii
import random

import json
import requests


def get_files_in_folder(folder_path):
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
                sub_list.append(item_path)
            elif not item.endswith('.txt'):
                vid_list.append(item_path)
            else:
                txt_list[appended_idx] = True

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)

        # If it's a directory, recursively call the function on that directory
        if os.path.isdir(item_path):
            new_subs, new_vids, new_txts = get_files_in_folder(item_path)
            sub_list.extend(new_subs)
            vid_list.extend(new_vids)
            txt_list.extend(new_txts)

    return sub_list, vid_list, txt_list


if __name__ == '__main__':

    current_dir = os.getcwd()
    print(os.getcwd())

    relative_root_folder_path = 'dataset/validation'
    root_folder_path = os.path.join(current_dir, relative_root_folder_path)

    # Call the function to get all files with the specified extension
    sub_path_list, vid_path_list, txt_list = get_files_in_folder(root_folder_path)

    vid_file_names = []

    for i in range(len(sub_path_list)):
        sub_path = sub_path_list[i]
        vid_path = vid_path_list[i]

        last_occurrence_index = vid_path.rfind('\\')
        vid_file_name = sub_path[last_occurrence_index + 1:]
        vid_file_names.append(vid_file_name)

        print(f"sub: {sub_path[-50:]} | vid: {vid_file_name}")

    input_subs = []
    correct_subs = []
    start_from = 46
    do_only_list = []
    do_transcription_list = []
    for i in range(len(txt_list)):
        if txt_list[i] is False:
            do_transcription_list.append(i)
            print("Doing transcription for: " + str(i))

    for i in range(start_from):
        input_subs.append(0)
        correct_subs.append(0)

    total_start_error = 0
    total_end_error = 0
    counter = 0
    for i in range(start_from, len(sub_path_list)):
        sub_path = sub_path_list[i]

        if len(do_only_list) > 0 and i not in do_only_list:
            input_subs.append(0)
            correct_subs.append(0)
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
            continue

        data = remove_non_ascii(data)
        sub = list(srt.parse(data))
        correct_subs.append(list(srt.parse(data)))

        offset = random.uniform(-360, 360)
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

        print(sub[0], " ; ", correct_sub[0])
        print(sub[-1], " ; ", correct_sub[-1])
        print(i, ": ", abs(sub[0].start.total_seconds() - correct_sub[0].start.total_seconds()), " , ",
              abs(sub[-1].start.total_seconds() - correct_sub[-1].start.total_seconds()))

        # Validation
        sub = input_subs[i]
        correct_sub = correct_subs[i]
        vid_path = vid_path_list[i]

        if i in do_transcription_list:
            save_mode = True
        else:
            save_mode = False

        speed, offset = get_fixed_speed_offset(sub, vid_path, save_mode=save_mode)
        print(f'Speed: {speed}, Offset: {offset}')

        if speed == 0:
            out_file = open("Output.txt", "a")
            out_file.write(str(i) + ': Failed ' + vid_file_names[i] + '\n')
            out_file.close()
            continue

        offset_time = datetime.timedelta(seconds=offset)

        for line in sub:
            line.start = speed * line.start + offset_time
            line.end = speed * line.end + offset_time

        sample_size = math.ceil(len(sub) * 0.1)

        sample_start_total = 0
        sample_end_total = 0

        for j in range(sample_size):
            sample_start_total += sub[j].start.total_seconds() - correct_sub[j].start.total_seconds()
            sample_end_total += sub[-j-1].start.total_seconds() - correct_sub[-j-1].start.total_seconds()

        start_error = sample_start_total / sample_size
        end_error = sample_end_total / sample_size

        print(sub[0], " ; ", correct_sub[0])
        print(sub[-1], " ; ", correct_sub[-1])
        print(i, ": ", abs(sub[0].start - correct_sub[0].start), " , ", abs(sub[-1].start - correct_sub[-1].start))

        #start_error = sub[0].start.total_seconds() - correct_sub[0].start.total_seconds()
        #end_error = sub[-1].start.total_seconds() - correct_sub[-1].start.total_seconds()

        out_file = open("Output.txt", "a")
        out_file.write(str(i) + ': ' + str(start_error) + ', ' +
                       str(end_error) + ' ' +
                       vid_file_names[i] + '\n')
        out_file.close()

        total_start_error += abs(start_error)
        total_end_error += abs(end_error)
        counter += 1

        # Export sub
        sub_out_path = os.path.join(current_dir, 'output_subs/' + str(i) + "_synced.srt")
        sub_out_file = open(sub_out_path, "w")
        sub_out_file.write(srt.compose(sub))
        sub_out_file.close()

    out_file = open("Output.txt", "a")
    out_file.write(f"Average start: {total_start_error / counter}\nAverage end: {total_end_error / counter}")

    out_file.close()

