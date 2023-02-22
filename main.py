import datetime
import os
import string

import srt
import numpy as np
from sklearn.linear_model import LinearRegression
from video_slicer import get_audio_clips
from audio_matcher import get_audio_timestamps


def remove_non_ascii(a_str):
    ascii_chars = set(string.printable)

    return ''.join(
        filter(lambda x: x in ascii_chars, a_str)
    )


if __name__ == '__main__':
    srt_path = "hotd2"
    video_path = os.path.join(os.path.dirname(__file__), "hotd2.mkv")

    f = open(srt_path + ".srt", "r")
    data = f.read()
    f.close()
    data = remove_non_ascii(data)
    subs = list(srt.parse(data))

    end_cut = 2*60

    debug_mode = False

    #audio_path = os.path.join(os.path.dirname(__file__), "hg2.mp3")

    #audio = load_audio(audio_path)
    print("Converting video to audio clips...")
    # TODO: Clean code
    audio1, audio2, second_start = get_audio_clips(video_path, section_length=12*60, end_cut=end_cut)
    if debug_mode:
        print(audio1.shape)
        print(audio1)

    not_enough_points_message = "Transcription failed: not enough anchor points."
    done_message_1 = "1/2 done"
    done_message_2 = "2/2 done"

    anchor_threshold = 2

    done = False
    sub_times1 = []
    sub_times2 = []
    transcript_times1 = []
    transcript_times2 = []
    attempt = 0
    while not done:
        if attempt == 0:
            print("Attempting fast transcription with default parameters...")
            sub_times1, transcript_times1 = get_audio_timestamps(audio1, subs, second_mode=False, offset=0)
            print(done_message_1)
            if len(sub_times1) < anchor_threshold:
                print(not_enough_points_message)
                attempt += 1
                continue
            sub_times2, transcript_times2 = get_audio_timestamps(audio2, subs, second_mode=True, offset=second_start)
            print(done_message_2)
            if len(sub_times1) < anchor_threshold:
                print(not_enough_points_message)
                attempt += 1
                continue
        elif attempt == 1:
            print("Attempting slower transcription...")
            sub_times1, transcript_times1 = get_audio_timestamps(audio1, subs, second_mode=False, offset=0,
                                                                 model_type="base.en")
            print(done_message_1)
            if len(sub_times1) < anchor_threshold:
                print(not_enough_points_message)
                attempt += 1
                continue
            sub_times2, transcript_times2 = get_audio_timestamps(audio2, subs, second_mode=True, offset=second_start,
                                                                 model_type="base.en")
            print(done_message_2)
            if len(sub_times1) < anchor_threshold:
                print(not_enough_points_message)
                attempt += 1
                continue
        elif attempt == 2:
            print("Attempting transcription with lower thresholds...")
            sub_times1, transcript_times1 = get_audio_timestamps(audio1, subs, second_mode=False, offset=0,
                                                                 model_type="base.en", min_ratio=70, minimum_length=20)
            print(done_message_1)
            if len(sub_times1) < anchor_threshold:
                print(not_enough_points_message)
                attempt += 1
                continue
            sub_times2, transcript_times2 = get_audio_timestamps(audio2, subs, second_mode=True, offset=second_start,
                                                                 model_type="base.en", min_ratio=70, minimum_length=20)
            print(done_message_2)
            if len(sub_times1) < anchor_threshold:
                print(not_enough_points_message)
                attempt += 1
                continue
        elif attempt == 3:
            print("Creating longer audio clips...")
            audio1, audio2, second_start = get_audio_clips(video_path, section_length=15*60, end_cut=end_cut)

            print("Attempting transcription...")
            sub_times1, transcript_times1 = get_audio_timestamps(audio1, subs, second_mode=False, offset=0,
                                                                 model_type="base.en", min_ratio=70, minimum_length=20)
            print(done_message_1)
            if len(sub_times1) < anchor_threshold:
                print(not_enough_points_message)
                attempt += 1
                continue
            sub_times2, transcript_times2 = get_audio_timestamps(audio2, subs, second_mode=True, offset=second_start,
                                                                 model_type="base.en", min_ratio=70, minimum_length=20)
            print(done_message_2)
            if len(sub_times1) < anchor_threshold:
                print(not_enough_points_message)
                attempt += 1
                continue
        else:
            print("Transcription failed: Try setting transcription parameters manually.")
            exit()
        done = True

    print("Transcription Successful!")

    print("Syncing the srt file...")
    sub_times = sub_times1 + sub_times2
    transcript_times = transcript_times1 + transcript_times2

    sub_ratio = len(sub_times2) / len(sub_times1)
    sub_ratio = 1

    sub_weights = np.array([sub_ratio] * len(sub_times1) + [1] * len(sub_times2))

    x = np.array(sub_times).reshape((-1, 1))
    y = np.array(transcript_times)

    model = LinearRegression()
    model.fit(x, y, sub_weights)

    a = model.coef_[0]
    b = model.intercept_

    if debug_mode:
        print(str(a) + ", " + str(b))

    for sub in subs:
        sub.start = a * sub.start + datetime.timedelta(seconds=b)
        sub.end = a * sub.end + datetime.timedelta(seconds=b)

    out_path = srt_path + "_synced.srt"
    out_file = open(out_path, "w")
    out_file.write(srt.compose(subs))
    out_file.close()

    print("Exported successfully as " + out_path + "!")
