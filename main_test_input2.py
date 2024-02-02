import datetime
import math
import os
import string

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


if __name__ == '__main__':

    current_dir = os.getcwd()
    print(os.getcwd())

    relative_root_folder_path = 'dataset/test'
    suffix = "correct"
    input_prefix = "k2"
    root_folder_path = os.path.join(current_dir, relative_root_folder_path)

    # Call the function to get all files with the specified extension
    sub_path_list, vid_path_list, txt_list, input_sub_path_list = get_files_in_folder(root_folder_path, suffix,
                                                                                      input_prefix)

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
    start_from = 0
    do_only_list = []
    do_transcription_list = []
    for i in range(len(txt_list)):
        if txt_list[i] is False:
            do_transcription_list.append(i)
            print("Doing transcription for: " + str(i))

    total_start_error = 0
    total_end_error = 0
    counter = 0
    for i in range(start_from, len(sub_path_list)):
        sub_path = sub_path_list[i]
        input_sub_path = input_sub_path_list[i]

        correct_data = None
        try:
            f = open(sub_path, "r", errors="ignore")
            correct_data = f.read()
            f.close()
        except:
            out_file = open("Output.txt", "a")
            out_file.write(str(i) + ": Failed - can't open " + vid_file_names[i] + '\n')
            out_file.close()
            continue

        input_data = None
        try:
            f = open(input_sub_path, "r", errors="ignore")
            input_data = f.read()
            f.close()
        except:
            out_file = open("Output.txt", "a")
            out_file.write(str(i) + ": Failed - can't open input " + vid_file_names[i] + '\n')
            out_file.close()
            continue

        correct_data = remove_non_ascii(correct_data)
        input_data = remove_non_ascii(input_data)
        sub = list(srt.parse(input_data))
        input_subs.append(sub)
        correct_subs.append(list(srt.parse(correct_data)))

        correct_sub = correct_subs[i]
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

        first_line = 0
        match_count = 0
        for j in range(len(sub)):
            if correct_sub[match_count].content == sub[j].content:
                if match_count == 0:
                    first_line = j
                match_count += 1
            else:
                match_count = 0
            if match_count == 3:
                break

        for j in range(sample_size):
            sample_start_total += sub[j+first_line].start.total_seconds() - correct_sub[j].start.total_seconds()
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

