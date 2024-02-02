import datetime
import os
import string

import srt
import numpy as np
from sklearn.linear_model import LinearRegression
from video_slicer import get_audio_clips
from audio_matcher import get_audio_timestamps

import json
import requests

def remove_non_ascii(a_str):
    ascii_chars = set(string.printable)

    return ''.join(
        filter(lambda x: x in ascii_chars, a_str)
    )


api_token = "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"

API_GPTJ_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-j-6B?wait_for_model=true"
API_GPT2_URL = "https://api-inference.huggingface.co/models/gpt2"
API_GPT_NEOX20B_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neox-20b"
headers = {"Authorization": f"Bearer {api_token}"}


def query_tmp(payload):
    data = json.dumps([payload] + ["I am very funny because"])
    response = requests.request("POST", API_GPTJ_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))


def query(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_GPTJ_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))


def semantic_match(sentence1, sentence2):
    payload = f"""
There is a program that generates audio transcription of speech. It was used to transcribe the sentence, and then it
was translated into serbian by a translation tool. It is possible that the original transcription contained some errors,
meaning that the translation may also contain some errors. There is also a second sentence provided
in serbian.
1) Transcribed + translated sentence: {sentence1}
2) Second sentence: {sentence2}
Please output True if the first sentence has the same meaning as the second sentence, taking into account that the first
one may contain some errors. Output True only if you are reasonably sure, otherwise, output False
The answer is (True/False):"""
    data = query({f"inputs": f"{payload}"})
    print(data)
    exit()


def linear_regression(list1, list2, weights):
    x = np.array(list1).reshape((-1, 1))
    y = np.array(list2)

    model = LinearRegression()
    model.fit(x, y, weights)

    return model.coef_[0], model.intercept_


def fix_subtitles(srt_path, video_path):
    f = open(srt_path, "r")
    data = f.read()
    f.close()
    data = remove_non_ascii(data)
    subs = list(srt.parse(data))

    a, b = get_fixed_speed_offset(subs, video_path)

    b_time = datetime.timedelta(seconds=b)

    for sub in subs:
        sub.start = a * sub.start + b_time
        sub.end = a * sub.end + b_time

    out_path = srt_path + "_synced.srt"
    out_file = open(out_path, "w")
    out_file.write(srt.compose(subs))
    out_file.close()

    print("Exported successfully as " + out_path + "!")


def get_fixed_speed_offset(subs, video_path):
    end_cut = 10 * 60

    debug_mode = True

    # audio_path = os.path.join(os.path.dirname(__file__), "hg2.mp3")

    # audio = load_audio(audio_path)
    print("Converting video to audio clips...")
    # TODO: Clean code
    audio1, audio2, second_start = get_audio_clips(video_path, section_length=15 * 60, end_cut=end_cut)
    if debug_mode:
        print(audio1.shape)
        print(audio1)

    not_enough_points_message = "Transcription failed: not enough anchor points."
    done_message_1 = "1/2 done"
    done_message_2 = "2/2 done"

    successful_first_half = False

    anchor_threshold = 10

    done = False
    sub_times1 = []
    sub_times2 = []
    transcript_times1 = []
    transcript_times2 = []
    attempt = 1
    while not done:
        if attempt == 0:
            print("Attempting fast transcription with default parameters...")
            sub_times1, transcript_times1 = get_audio_timestamps(audio1, subs, second_mode=False, offset=0)
            print(done_message_1)
            if len(sub_times1) < anchor_threshold:
                print(not_enough_points_message)
                attempt += 1
                continue
            successful_first_half = True
            sub_times2, transcript_times2 = get_audio_timestamps(audio2, subs, second_mode=True, offset=second_start)
            print(done_message_2)
            if len(sub_times2) < anchor_threshold:
                print(not_enough_points_message)
                attempt += 1
                continue
        elif attempt == 1:
            print("Attempting slower transcription...")
            if successful_first_half is False:
                sub_times1, transcript_times1 = get_audio_timestamps(audio1, subs, second_mode=False, offset=0,
                                                                     model_type="small.en")
                print(done_message_1)
                if len(sub_times1) < anchor_threshold:
                    print(not_enough_points_message)
                    attempt += 1
                    continue
                successful_first_half = True
            sub_times2, transcript_times2 = get_audio_timestamps(audio2, subs, second_mode=True, offset=second_start,
                                                                 model_type="small.en")
            print(done_message_2)
            if len(sub_times2) < anchor_threshold:
                print(not_enough_points_message)
                attempt += 1
                continue
        elif attempt == 2:
            print("Attempting transcription with lower thresholds...")
            if successful_first_half is False:
                sub_times1, transcript_times1 = get_audio_timestamps(audio1, subs, second_mode=False, offset=0,
                                                                     model_type="small.en", min_ratio=80, minimum_length=0)
                print(done_message_1)
                if len(sub_times1) < anchor_threshold:
                    print(not_enough_points_message)
                    attempt += 1
                    continue
                successful_first_half = True
            sub_times2, transcript_times2 = get_audio_timestamps(audio2, subs, second_mode=True, offset=second_start,
                                                                 model_type="small.en", min_ratio=80, minimum_length=0)
            print(done_message_2)
            if len(sub_times2) < anchor_threshold:
                print(not_enough_points_message)
                attempt += 1
                continue
        elif attempt == 3:
            print("Creating longer audio clips...")
            audio1, audio2, second_start = get_audio_clips(video_path, section_length=45 * 60, end_cut=end_cut)

            print("Attempting transcription...")
            if successful_first_half is False:
                sub_times1, transcript_times1 = get_audio_timestamps(audio1, subs, second_mode=False, offset=0,
                                                                     model_type="small.en", min_ratio=80, minimum_length=0,
                                                                     skip_percent=0.3)  # Hardcoded
                print(done_message_1)
                if len(sub_times1) < anchor_threshold:
                    print(not_enough_points_message)
                    attempt += 1
                    continue
                successful_first_half = True
            sub_times2, transcript_times2 = get_audio_timestamps(audio2, subs, second_mode=True, offset=second_start,
                                                                 model_type="small.en", min_ratio=80, minimum_length=0,
                                                                 skip_percent=0.3)
            print(done_message_2)
            if len(sub_times2) < anchor_threshold:
                print(not_enough_points_message)
                attempt += 1
                continue
        else:
            print("Transcription failed: Try setting transcription parameters manually.")
            return 0, 0
            exit()
        done = True

    print("Transcription Successful!")

    print("Syncing the srt file...")
    sub_times = sub_times1 + sub_times2
    transcript_times = transcript_times1 + transcript_times2

    sub_ratio = len(sub_times2) / len(sub_times1)
    # sub_ratio = 1

    sub_weights = np.array([sub_ratio] * len(sub_times1) + [1] * len(sub_times2))

    x = np.array(sub_times).reshape((-1, 1))
    y = np.array(transcript_times)

    model = LinearRegression()
    model.fit(x, y, sub_weights)

    a = model.coef_[0]
    b = model.intercept_

    if debug_mode:
        print(str(a) + ", " + str(b))

    # Filtering outliers:
    acceptable_error = 0.15
    average_error = -1
    unfiltered_sub_times = sub_times
    unfiltered_transcript_times = transcript_times
    removed_point = False
    while average_error == -1 or (average_error > acceptable_error and removed_point):
        removed_point = False
        filtered_sub_times = []
        filtered_transcript_times = []
        sub_len1 = 0
        sub_len2 = 0
        error_sum = 0
        for i in range(len(unfiltered_sub_times)):
            error_sum += abs(unfiltered_transcript_times[i] - (a * unfiltered_sub_times[i] + b))
        average_error = error_sum / len(unfiltered_sub_times)
        error_threshold = 2 * average_error
        for i in range(len(unfiltered_sub_times)):
            error = abs(unfiltered_transcript_times[i] - (a * unfiltered_sub_times[i] + b))
            if error < error_threshold:
                print("Keeping: " + str(error))
                filtered_sub_times.append(unfiltered_sub_times[i])
                filtered_transcript_times.append(unfiltered_transcript_times[i])
                if i < len(sub_times1):
                    sub_len1 += 1
                else:
                    sub_len2 += 1
            else:
                print("Error: " + str(error))
                removed_point = True

        if len(filtered_sub_times) > 0 and sub_len1 > 0 and sub_len2 > 0:
            sub_ratio = sub_len2 / sub_len1
            sub_weights = np.array([sub_ratio] * sub_len1 + [1] * sub_len2)
            a, b = linear_regression(filtered_sub_times, filtered_transcript_times, sub_weights)
        else:
            break

        unfiltered_sub_times = filtered_sub_times
        unfiltered_transcript_times = filtered_transcript_times
        print("End of loop")

    return a, b


if __name__ == '__main__':
    srt_path = "hg1.srt"
    video_path = os.path.join(os.path.dirname(__file__), "hg.mp4")
    fix_subtitles(srt_path, video_path)
