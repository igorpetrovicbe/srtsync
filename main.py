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

    in_file = "hg2"

    f = open(in_file + ".srt", "r")
    data = f.read()
    f.close()
    data = remove_non_ascii(data)
    subs = list(srt.parse(data))

    print(subs[0].content + "\n")
    print(subs[1].content + "\n")
    print(subs[2].content + "\n")
    print(str(subs[0]) + "\n")
    print(str(subs[1]) + "\n")
    print(str(subs[2]) + "\n")

    #audio_path = os.path.join(os.path.dirname(__file__), "hg2.mp3")
    video_path = os.path.join(os.path.dirname(__file__), "hg.mp4")

    #audio = load_audio(audio_path)
    audio1, audio2, second_start = get_audio_clips(video_path, section_length=10*60, end_cut=10*60)
    print(audio1.shape)
    print(audio1)

    sub_times1, transcript_times1 = get_audio_timestamps(audio1, subs, second_mode=False, offset=0)
    sub_times2, transcript_times2 = get_audio_timestamps(audio2, subs, second_mode=True, offset=second_start)

    sub_times = sub_times1 + sub_times2
    transcript_times = transcript_times1 + transcript_times2

    x = np.array(sub_times).reshape((-1, 1))
    y = np.array(transcript_times)

    model = LinearRegression()
    model.fit(x, y)

    a = model.coef_[0]
    b = model.intercept_
    print(str(a) + ", " + str(b))

    for sub in subs:
        sub.start = a * sub.start + datetime.timedelta(seconds=b)
        sub.end = a * sub.end + datetime.timedelta(seconds=b)

    out_file = open(in_file + "_out.srt", "w")
    out_file.write(srt.compose(subs))
    out_file.close()
