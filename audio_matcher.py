import os

import torch
import whisper
from fuzzywuzzy import fuzz


def get_audio_timestamps(audio, subs, second_mode, offset=0, skip_percent=0.5, model_type="tiny.en", min_ratio=80,
                         minimum_length=30):
    debug_mode = False

    if debug_mode:
        print(torch.cuda.is_available())

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if debug_mode:
        print("Using device: " + device)

    model = whisper.load_model(model_type).to(device)
    sub_sentences = []

    sentence = ""
    for sub in subs:
        i = 0
        while i < len(sub.content):
            char = sub.content[i]
            if char == '.' or char == '?' or char == '!':
                sentence = sentence.replace('\n', ' ')
                end_time = (sub.end - sub.start) * i / len(sub.content) + sub.start
                sub_sentences.append((sentence[:], end_time.total_seconds()))
                if False and debug_mode:
                    print(sentence)
                sentence = ""
                while i + 1 < len(sub.content):
                    if sub.content[i + 1] == '.' or sub.content[i + 1] == '?' or sub.content[i + 1] == '!':
                        i = i + 1
                    else:
                        break
            else:
                sentence += char
            i += 1

    result = model.transcribe(audio, language="en", max_initial_timestamp=None)
    if debug_mode:
        print(result)
    segments = result["segments"]

    transcript_sentences = []
    sentence = ""

    for segment in segments:
        i = 0
        text = segment["text"]
        while i < len(text):
            char = text[i]
            if char == '.' or char == '?' or char == '!':
                sentence = sentence.replace('\n', ' ')
                end_time = (segment["end"] - segment["start"]) * i / len(text) + segment["start"] + offset
                transcript_sentences.append((sentence[:], end_time))
                if debug_mode:
                    print(sentence)
                sentence = ""
                while i + 1 < len(text):
                    if segment["text"][i + 1] == '.' or segment["text"][i + 1] == '?' or segment["text"][i + 1] == '!':
                        i = i + 1
                    else:
                        break
            else:
                sentence += char
            i += 1

    start = 0
    max_span = 200
    sub_times = []
    if second_mode:
        start_offset_sub = int(len(sub_sentences) * skip_percent)  # Not in use for now
    else:
        start_offset_sub = 0
    # start_offset_trans = int(len(transcript_sentences) * skip_percent)  # Not in use for now
    last = 0
    transcript_times = []
    for s in range(start_offset_sub, len(sub_sentences)):
        sub_sentence = sub_sentences[s]
        if len(sub_sentence[0]) < minimum_length:
            if (start != 0 or not second_mode) and last < len(transcript_sentences) and len(transcript_sentences[last][0]) < minimum_length:
                last += 1
            continue
        i = last
        while i < min(last + max_span, len(transcript_sentences)) or (i < len(transcript_sentences) and start == 0 and second_mode):
            transcript_sentence = transcript_sentences[i]
            ratio = fuzz.ratio(sub_sentence[0], transcript_sentence[0])
            if ratio > min_ratio:
                if debug_mode:
                    print(sub_sentence[0] + " <-- " + transcript_sentence[0])
                sub_times.append(sub_sentence[1])
                transcript_times.append(transcript_sentence[1])
                last = i
                if start == 0:
                    start = i
                break
            i += 1

    return sub_times, transcript_times
