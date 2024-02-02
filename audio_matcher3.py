import os

import torch
import stable_whisper
from fuzzywuzzy import fuzz
import gc

def get_audio_timestamps(audio, subs, second_mode, save_mode, video_path, offset=0, skip_percent=0.5, model_type="base.en", min_ratio=80,
                         minimum_length=30):
    debug_mode = True

    if debug_mode:
        print(torch.cuda.is_available())

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if debug_mode:
        print("Using device: " + device)

    model = None
    if save_mode:
        try:
            model = stable_whisper.load_model(model_type).to(device)
        except:
            print("Failed to load model. Trying again with CPU.")
            device = "cpu"
            model = stable_whisper.load_model(model_type).to(device)

    sub_sentences = []

    sentence = ""
    sub_start = None
    from_start = True
    for sub in subs:
        i = 0
        if sub_start is None:
            sub_start = sub.start
            from_start = True
        while i < len(sub.content):
            char = sub.content[i]
            if char == '.' or char == '?' or char == '!':
                sentence = sentence.replace('\n', ' ')
                if from_start:
                    sub_sentences.append((sentence[:], sub_start.total_seconds()))
                if False and debug_mode:
                    print(sentence)
                sentence = ""
                sub_start = None
                while i + 1 < len(sub.content):
                    if sub.content[i + 1] == '.' or sub.content[i + 1] == '?' or sub.content[i + 1] == '!':
                        i = i + 1
                    else:
                        sub_start = (sub.end - sub.start) * i / len(sub.content) + sub.start
                        from_start = False
                        break
            else:
                sentence += char
            i += 1

    path_ext_index = video_path.rfind('.')
    transcript_file_path = video_path[0:path_ext_index] + '.txt'
    transcript_sentences = []
    if save_mode:
        result = model.transcribe(audio, language="en", max_initial_timestamp=None)

        model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()

        #if debug_mode:
        #    print(result)
        segments = result["segments"]

        if debug_mode:
            for segment in segments:
                print(segment["word_timestamps"])

        transcript_sentences = []
        sentence = ""
        start_time = None
        from_start = True
        for segment in segments:
            #transcript_sentences.append((segment["text"], segment["start"]))
            #continue
            i = 0
            k = 0
            text = segment["text"]
            timestamps = segment["word_timestamps"]
            word_text = ""
            if start_time is None:
                start_time = segment["start"]
                from_start = True
            while i < len(text):
                char = text[i]
                if len(word_text) < len(sentence) and k < len(timestamps):
                    word_text += timestamps[k]["word"]
                    if debug_mode:
                        print("WT: " + word_text + ", k = " + str(k))
                    k += 1
                if char == '.' or char == '?' or char == '!':
                    sentence = sentence.replace('\n', ' ')
                    # end_time = (segment["end"] - segment["start"]) * i / len(text) + segment["start"] + offset

                    ##if k < len(timestamps):
                    #    end_time = timestamps[k - 1]["timestamp"] + offset
                    #else:
                    #    end_time = segment["end"] + offset
                    #if from_start:
                    transcript_sentences.append((sentence[:], start_time))

                    start_time = None
                    if debug_mode:
                        print(sentence)
                    sentence = ""
                    while i + 1 < len(text):
                        if segment["text"][i + 1] == '.' or segment["text"][i + 1] == '?' or segment["text"][i + 1] == '!':
                            i = i + 1
                        else:
                            start_time = timestamps[k - 1]["timestamp"] + offset
                            from_start = False
                            break
                else:
                    sentence += char
                i += 1
            if len(sentence) > 0:
                transcript_sentences.append((sentence[:], start_time))
            start_time = None
            sentence = ""

        out_file = open(transcript_file_path, "w")
        for transcript_sentence in transcript_sentences:
            try:
                out_file.write(transcript_sentence[0] + ';' + str(transcript_sentence[1]) + '\n')
            except:
                print("Failed to write line")
        out_file.close()
    else:
        file = open(transcript_file_path, 'r')
        lines = file.readlines()

        count = 0
        # Strips the newline character
        for line in lines:
            count += 1
            split_index = line.rfind(';')
            sentence = line[0:split_index]
            start = float(line[split_index+1:])
            transcript_sentences.append((sentence, start))
            # print(sentence + '  -  ' + str(start))

    print("Begins...")

    filtered_sub_sentences = []
    filtered_transcript_sentences = []

    for sentence in sub_sentences:
        if len(sentence[0]) >= minimum_length:
            filtered_sub_sentences.append(sentence)

    for sentence in transcript_sentences:
        if len(sentence[0]) >= minimum_length:
            filtered_transcript_sentences.append(sentence)

    sub_times = []
    transcript_times = []
    for s in range(len(filtered_sub_sentences)):
        sub_sentence = filtered_sub_sentences[s]
        i = 0
        match_index = None
        found_double_match = False
        while i < len(filtered_transcript_sentences):
            transcript_sentence = filtered_transcript_sentences[i]
            ratio = fuzz.ratio(sub_sentence[0], transcript_sentence[0])
            if ratio > min_ratio:
                if debug_mode:
                    print(sub_sentence[0] + " <-- " + transcript_sentence[0] + ' ; Time: ' + str(transcript_sentence[1]))
                if match_index is not None:
                    found_double_match = True
                    break
                match_index = i
            i += 1
        if match_index is not None and not found_double_match:
            sub_times.append(sub_sentence[1])
            transcript_times.append(filtered_transcript_sentences[match_index][1])

    return sub_times, transcript_times, sub_sentences, transcript_sentences


def get_more_audio_timestamps(sub_sentences, transcript_sentences, a, b, max_time_diff):
    min_ratio = 80
    minimum_length = 5

    debug_mode = True

    sub_times = []
    transcript_times = []
    start_from = 0
    for s in range(0, len(sub_sentences)):
        sub_sentence = sub_sentences[s]
        match_index = None
        if len(sub_sentence[0]) < minimum_length:
            continue
        found_double_match = False
        for t in range(start_from, len(transcript_sentences)):
            transcript_sentence = transcript_sentences[t]
            new_sub_time = a * sub_sentence[1] + b
            if new_sub_time + max_time_diff < transcript_sentence[1]:
                break
            if abs(new_sub_time - transcript_sentence[1]) > max_time_diff:
                start_from = t
                continue
            ratio = fuzz.ratio(sub_sentence[0], transcript_sentence[0])
            if ratio > min_ratio:
                if debug_mode:
                    print(sub_sentence[0] + " <-- " + transcript_sentence[0] + ' ; Time: ' + str(transcript_sentence[1]))
                if match_index is not None:
                    found_double_match = True
                    break
                match_index = t

        if match_index is not None and not found_double_match:
            sub_times.append(sub_sentence[1])
            transcript_times.append(transcript_sentences[match_index][1])

    return sub_times, transcript_times
