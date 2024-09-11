import os

def ensure_path_exist(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


VID_DIR = "fs/vid"
AUDIO_DIR = "fs/aud"
LOOPS_DIR = "fs/loop"

def extract_audio_cmd(file_name: str):
    no_ext_file_name = ".".join(file_name.split(".")[0:-1])
    in_file = f"./{VID_DIR}/{file_name}"
    out_file = f"./{AUDIO_DIR}/{no_ext_file_name}.wav"
    if os.path.exists(out_file):
        return None


    plain_cmd = f"ffmpeg -i {in_file} -vn -y {out_file}"
    array_cmd = plain_cmd.split(" ")
    print(array_cmd)
    return array_cmd


import subprocess

def init():
    ensure_path_exist(AUDIO_DIR)
    ensure_path_exist(LOOPS_DIR)


def vid_2_audio():
    in_files = os.listdir(VID_DIR)
    proc_cmds = [ extract_audio_cmd(file) for file in in_files ]
    proc_cmds = list(filter(lambda cmd: cmd is not None, proc_cmds))

    for cmd in proc_cmds:
        subprocess.run(cmd)

    print(f"+++ total cmds run: {len(proc_cmds)}")

import wave
import numpy as np
import matplotlib.pyplot as plt

def generate_fade_signal(sample_num, dir="up"):
    start_value = 1
    end_value = 4

    if dir != "up":
        end_value = 1
        start_value = 4

    ramp = np.linspace(start_value, end_value, sample_num+1)[:-1]
    ramp = np.log2(ramp)/2

    ramp = np.expand_dims(ramp, 0)
    ramp = np.vstack((ramp, ramp))
    return ramp

def loop_audio(np_audio: np.ndarray, overlap_len):
    begin = np_audio[:, 0:overlap_len]
    rise_of_begin = generate_fade_signal(overlap_len, dir="up")
    begin = (begin*rise_of_begin).astype(np.int16)

    mid = np_audio[:, overlap_len:-overlap_len]
    end = np_audio[:, -overlap_len:]
    fall_of_end = generate_fade_signal(overlap_len, dir="down")
    end = (end*fall_of_end).astype(np.int16)

    crosfade = end+begin
    np_loop = np.hstack((mid,crosfade))
    loop_len = np_loop.shape[1]

    return np_loop, loop_len

def extract_loop(in_file, out_file):
    print(f"+++ opening {in_file}")
    in_wav = wave.open(in_file, "r")

    # stats
    wav_fs = in_wav.getframerate()
    wav_ch = in_wav.getnchannels()
    wav_smp_w = in_wav.getsampwidth()
    wav_len = in_wav.getnframes()

    wav_len_s = wav_len/wav_fs

    crossfade_len_s = 10
    src_data_len_s = 45
    overlap_len = crossfade_len_s*wav_fs
    loop_data_len = src_data_len_s*wav_fs
    if wav_len_s < (src_data_len_s + 1):
        print("+++ skip this sample, too short")
        return
    
    frame_offset = int((wav_len - loop_data_len)/2)
    _ = in_wav.readframes(frame_offset)
    src_byte_data = in_wav.readframes(loop_data_len)
    in_wav.close()


    np_data = np.frombuffer(src_byte_data, np.int16).reshape((2,loop_data_len))
    np_loop, loop_len = loop_audio(np_data, overlap_len)
    dst_byte_data = np_loop.flatten("F").tobytes()


    # write loop
    print(f"+++ saving {out_file}")
    out_wav = wave.open(out_file, "w")
    out_wav.setframerate(wav_fs)
    out_wav.setnchannels(wav_ch)
    out_wav.setsampwidth(wav_smp_w)
    out_wav.setnframes(loop_len)
    out_wav.writeframes(dst_byte_data)
    out_wav.close()

def audio_2_loops():
    in_files = os.listdir(AUDIO_DIR)
    full_in_files = [ os.path.join(AUDIO_DIR,in_file) for in_file in in_files ]
    full_out_files = [ os.path.join(LOOPS_DIR, in_file) for in_file in in_files ]

    for in_file, out_file in zip(full_in_files, full_out_files):
        extract_loop(in_file, out_file)

def main():
    init()
    vid_2_audio()
    audio_2_loops()

    
main()