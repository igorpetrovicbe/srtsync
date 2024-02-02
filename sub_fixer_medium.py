import datetime
import math
import os
import string

import srt
import numpy as np
from sklearn.linear_model import LinearRegression
from video_slicer import get_audio_clips, get_single_audioclip
from audio_matcher3 import get_audio_timestamps, get_more_audio_timestamps

import json
import requests
import matplotlib.pyplot as plt


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


def draw_plot(filtered_sub_times, filtered_transcript_times, a, b, msg):
    # Generate predicted Y values for the line
    x_hat = np.linspace(min(filtered_sub_times), max(filtered_sub_times), 100)  # Generate 100 points for the x-axis
    y_hat = a * x_hat + b

    X = np.array(filtered_sub_times)
    Y = np.array(filtered_transcript_times)

    # Create a scatter plot for the data points
    plt.scatter(X, Y, label='Referentne tačke ' + msg)

    # Plot the line y_hat = k*x_hat + n
    plt.plot(x_hat, y_hat, label=f'Transformacija: f(t) = {a:.2f} * t + {b:.2f}', color='red')

    # Add labels and a legend
    plt.xlabel('Trenuci titla')
    plt.ylabel('Trenuci transkripcije')
    plt.legend()

    # Display the plot
    plt.show()


def calculate_offsets(sub_times, transcript_times, debug_mode=False):
    sub_weights = np.array([1] * len(sub_times))

    x = np.array(sub_times).reshape((-1, 1))
    y = np.array(transcript_times)

    model = LinearRegression()
    model.fit(x, y, sub_weights)

    a = model.coef_[0]
    b = model.intercept_

    if debug_mode:
        print(str(a) + ", " + str(b))

    draw_plot(sub_times, transcript_times, a, b, '(pre uklanjanja izuzetaka)')

    # Filtering outliers:
    acceptable_error = 0.15
    average_error = -1
    unfiltered_sub_times = sub_times
    unfiltered_transcript_times = transcript_times
    removed_point = False
    filtered_sub_times = []
    filtered_transcript_times = []
    while average_error == -1 or (average_error > acceptable_error and removed_point):
        removed_point = False
        filtered_sub_times = []
        filtered_transcript_times = []
        sub_len = 0
        error_sum = 0
        for i in range(len(unfiltered_sub_times)):
            error_sum += abs(unfiltered_transcript_times[i] - (a * unfiltered_sub_times[i] + b))
        average_error = error_sum / len(unfiltered_sub_times)
        error_threshold = max(0.1, 3 * average_error)
        for i in range(len(unfiltered_sub_times)):
            error = abs(unfiltered_transcript_times[i] - (a * unfiltered_sub_times[i] + b))
            if error < error_threshold:
                print("Keeping: " + str(error))
                filtered_sub_times.append(unfiltered_sub_times[i])
                filtered_transcript_times.append(unfiltered_transcript_times[i])
                sub_len += 1
            else:
                print("Error: " + str(error))
                removed_point = True

        if len(filtered_sub_times) > 0 and sub_len > 0:
            sub_weights = np.array([1] * sub_len)
            a, b = linear_regression(filtered_sub_times, filtered_transcript_times, sub_weights)
        else:
            break

        unfiltered_sub_times = filtered_sub_times
        unfiltered_transcript_times = filtered_transcript_times
        print("End of loop")

    draw_plot(filtered_sub_times, filtered_transcript_times, a, b, '(nakon uklanjanja izuzetaka)')

    return a, b, filtered_sub_times, filtered_transcript_times


def fix_subtitles(srt_path, video_path):
    f = open(srt_path, "r")
    data = f.read()
    f.close()
    data = remove_non_ascii(data)
    subs = list(srt.parse(data))

    a, b = get_fixed_speed_offset(subs, video_path, save_mode=False)

    b_time = datetime.timedelta(seconds=b)

    for sub in subs:
        sub.start = a * sub.start + b_time
        sub.end = a * sub.end + b_time

    out_path = srt_path + "_synced.srt"
    out_file = open(out_path, "w")
    out_file.write(srt.compose(subs))
    out_file.close()

    print("Exported successfully as " + out_path + "!")


def get_fixed_speed_offset(subs, video_path, save_mode):
    end_cut = 10 * 60

    debug_mode = True

    # audio_path = os.path.join(os.path.dirname(__file__), "hg2.mp3")

    # audio = load_audio(audio_path)
    print("Converting video to audio clips...")
    # TODO: Clean code

    if save_mode:
        audio = get_single_audioclip(video_path)
        if audio is None:
            return 0, 0
        if debug_mode:
            print(audio.shape)
    else:
        audio = []
    not_enough_points_message = "Transcription failed: not enough anchor points."
    done_message_1 = "1/2 done"
    done_message_2 = "2/2 done"

    successful_first_half = False

    anchor_threshold = 10

    done = False
    sub_times = []
    transcript_times = []
    sub_sentences = []
    transcript_sentences = []
    attempt = 1
    while not done:
        if attempt == 0:
            print("Attempting fast transcription with default parameters...")
            sub_times, transcript_times, sub_sentences, transcript_sentences = get_audio_timestamps(audio, subs, second_mode=False, offset=0,
                                                               save_mode=save_mode, video_path=video_path)
            print(done_message_1)
            if len(sub_times) < anchor_threshold:
                print(not_enough_points_message)
                attempt += 1
                continue
        elif attempt == 1:
            print("Attempting slower transcription...")
            sub_times, transcript_times, sub_sentences, transcript_sentences = get_audio_timestamps(audio, subs, second_mode=False, offset=0,
                                                               model_type="small.en", save_mode=save_mode,
                                                               video_path=video_path)
            print(done_message_1)
            if len(sub_times) < anchor_threshold:
                print(not_enough_points_message)
                # attempt += 1
                attempt += 2
                save_mode = False
                continue
        elif attempt == 2:
            print("Attempting transcription with lower thresholds...")
            sub_times, transcript_times, sub_sentences, transcript_sentences = get_audio_timestamps(audio, subs, second_mode=False, offset=0,
                                                               model_type="small.en", min_ratio=75, minimum_length=20,
                                                               save_mode=save_mode, video_path=video_path)
            print(done_message_1)
            if len(sub_times) < anchor_threshold:
                print(not_enough_points_message)
                attempt += 1
                continue
        elif attempt == 3:
            print("Attempting transcription with a larger model...")
            sub_times, transcript_times, sub_sentences, transcript_sentences = get_audio_timestamps(audio, subs, second_mode=False, offset=0,
                                                               model_type="small.en", min_ratio=75, minimum_length=20,
                                                               save_mode=save_mode, video_path=video_path)
            print(done_message_1)
            if len(sub_times) < anchor_threshold:
                print(not_enough_points_message)
                attempt += 1
                continue
        elif attempt == 4:
            if len(sub_times) < 2:
                print(not_enough_points_message)
                attempt += 1
                continue
        else:
            print("Transcription failed: Try setting transcription parameters manually.")
            return 0, 0
        done = True

    print("Transcription Successful!")

    print("Syncing the srt file...")

    a, b, filtered_sub_times, filtered_transcript_times = calculate_offsets(sub_times, transcript_times, debug_mode=True)

    # PASS 1

    sub_times, transcript_times = get_more_audio_timestamps(sub_sentences, transcript_sentences, a, b, max_time_diff=4)

    if len(sub_times) < 2:
        return a, b

    a, b, filtered_sub_times, filtered_transcript_times = calculate_offsets(sub_times, transcript_times, debug_mode=True)

    # PASS 2

    sub_times, transcript_times = get_more_audio_timestamps(sub_sentences, transcript_sentences, a, b, max_time_diff=2)

    if len(sub_times) < 2:
        return a, b

    a, b, filtered_sub_times, filtered_transcript_times = calculate_offsets(sub_times, transcript_times,
                                                                            debug_mode=True)

    # PASS 3

    sub_times, transcript_times = get_more_audio_timestamps(sub_sentences, transcript_sentences, a, b, max_time_diff=1)

    if len(sub_times) < 2:
        return a, b

    a, b, filtered_sub_times, filtered_transcript_times = calculate_offsets(sub_times, transcript_times,
                                                                            debug_mode=True)

    if len(filtered_sub_times) > 1:
        sample_size = math.ceil(len(filtered_sub_times) * 0.05)

        final_sub_times = filtered_sub_times[0:sample_size] + filtered_sub_times[-sample_size:]
        final_transcript_times = filtered_transcript_times[0:sample_size] + filtered_transcript_times[-sample_size:]

        # final_sub_times = [filtered_sub_times[0]] + [filtered_sub_times[-1]]
        # final_transcript_times = [filtered_transcript_times[0]] + [filtered_transcript_times[-1]]

        weights = np.array([1] * len(final_sub_times))
        a, b = linear_regression(final_sub_times, final_transcript_times, weights)

    return a, b


if __name__ == '__main__':
    srt_path = "hg1.srt"
    video_path = os.path.join(os.path.dirname(__file__), "hg.mp4")
    fix_subtitles(srt_path, video_path)
