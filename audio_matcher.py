import os

import torch
import whisper
from fuzzywuzzy import fuzz


def get_audio_timestamps(audio, subs, second_mode, offset, skip_percent=3 / 4):
    print(torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(device)
    model = whisper.load_model("tiny").to(device)
    video_path = os.path.join(os.path.dirname(__file__), "hg.mp4")

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

    result = model.transcribe(audio)
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
    last = start
    minimum_length = 20
    max_span = 40
    min_ratio = 70
    skipped = 0
    sub_times = []
    start_offset = len(sub_sentences) * skip_percent
    transcript_times = []
    for s in range(len(sub_sentences)):
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
                print(sub_sentence[0] + " <-- " + transcript_sentence[0])
                sub_times.append(sub_sentence[1])
                transcript_times.append(transcript_sentence[1])
                last = i
                if start == 0:
                    start = i
                break
            i += 1

    return sub_times, transcript_times
