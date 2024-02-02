import numpy as np
import moviepy.editor as mp


def get_audio_clips(path, section_length=10*60, end_cut=0):
    clip = mp.VideoFileClip(r'{}'.format(path))
    clip_end = clip.duration - end_cut

    section_length = min(section_length, clip_end / 2)

    clip1 = clip.subclip(0, section_length)
    clip2 = clip.subclip(clip_end - section_length, clip_end)

    audio1_array = clip1.audio.to_soundarray(fps=16000).astype(np.float32)
    audio2_array = clip2.audio.to_soundarray(fps=16000).astype(np.float32)

    # clip1.audio.write_audiofile(r'{}mp3'.format(path[:-4] + "1." ), fps=16000)

    return np.mean(audio1_array, axis=1), np.mean(audio2_array, axis=1), clip_end - section_length


def get_single_audioclip(path):
    clip = mp.VideoFileClip(r'{}'.format(path))

    try:
        audio_array = clip.audio.to_soundarray(fps=16000).astype(np.float32)
    except:
        print("Moviepy audio conversion failed")
        return None

    return np.mean(audio_array, axis=1)
