from pydualsense import *
import time
import math

import sounddevice as sd
import keyboard

import numpy as np
import re

def clamp(minimum, x, maximum):
    return max(minimum, min(x, maximum))

DIR = {"up": 1, "down": -1}
def safe_dir(dir):
    if dir in DIR:
        return DIR[dir]
    return 1


class haptic_control():
    def __init__(self):
        self.amplitude = 0.1
        self.amp_step = 0.01

        self.freq_target = 60
        self.freq_step = 20
        self.last_freq = self.freq_target
        self.next_freq = self.freq_target
        self.freq_actual = self.freq_target
        self.freq_ramp

        self.phase_offset = 0
        self.phase_Step = (2 * np.pi)/64

        self.modable = [self.amp_mod, self.freq_mod, self.phase_offset_mod]

        self.signal_0_phase = 0
        self.signal_1_phase = 0

        self.fs = 0
        self.transition_time = 0.5
        self.transition_frames = 0
        self.transition_at = 0
        self.transition_left = 0
        

    def set_fs(self, fs):
        self.fs = fs
        self.transition_frames = self.fs * self.transition_time



    def amp_mod(self, dir):
        self.amplitude += self.amp_step * safe_dir(dir)
        self.amplitude = clamp(0, self.amplitude, 1)

        print(f"amp {self.amplitude:f}")

    def freq_mod(self, dir):
        self.freq_target += self.freq_step * safe_dir(dir)
        self.freq_target = clamp(30, self.freq_target, 600)
        print(f"freq {self.freq_target:f}")

        self.transition_at = 0
        self.transition_left = self.transition_frames

    def phase_offset_mod(self, dir):
        self.phase_offset += self.phase_Step * safe_dir(dir)
        self.phase_offset = clamp(0, self.phase_offset, 2 * np.pi)
        print(f"phase_offset {self.phase_offset:f}")

    def act(self, *args):
        args[0](args[1])

    def freq_synth(self, frames):

        self.transition_frames - self.transition_at

        if self.transition_at == 0:
            ramp = np.linspace(self.last_freq, self.next_freq, self.transition_frames)




    

hpc = haptic_control()

def safe_exit_wrapper(fn):
    try:
        fn()
    except KeyboardInterrupt:
        print("CTRL+C pressed")

def _inputs_loop(audio_stream: sd.OutputStream):
    global hpc

    pairs = ["qa", "ws", "ed", "rf", "tg", "yh", "uj", "ik", "ol"]
    act_dirs = ["up", "down"]

    for pair, fn in zip(pairs, hpc.modable):
        for key, act_dir in zip(pair, act_dirs):
            keyboard.add_hotkey(key, hpc.act, args=([fn, act_dir]))

    print("from inputs loop")
    with audio_stream:
        keyboard.wait()


def inputs_loop(audio_stream: sd.OutputStream):
    wrap = lambda: _inputs_loop(audio_stream)
    safe_exit_wrapper(wrap)


def dsloop(audio_stream: sd.OutputStream):


    # get dualsense instance
    dualsense = pydualsense()
    dualsense.init()

    print('Trigger Effect demo started')


    dualsense.triggerL.setMode(TriggerModes.Rigid)
    dualsense.triggerL.setForce(1, 255)

    dualsense.triggerR.setMode(TriggerModes.Pulse_A)
    dualsense.triggerR.setForce(0, 200)
    dualsense.triggerR.setForce(1, 255)
    dualsense.triggerR.setForce(2, 175)

    # loop until r1 is pressed to feel effect


    phase = 0
    cycle = math.pi * 2
    samples = 100
    td = 1 / samples
    delta = cycle / samples


    ts = time.perf_counter()
    with audio_stream:
        while not dualsense.state.R1:
            now = time.perf_counter()
            if now - ts > td:
                ts = now
                phase += delta
                value = ((math.sin(phase)+1)/2)*255
                value = int(value)
                dualsense.setLeftMotor(value)
                
            
    # terminate the thread for message and close the device
    dualsense.close()

def find_dualsence_audio_output():
    devices = sd.query_devices()
    name_query = lambda dev: dev["name"].find("DualSense") > -1
    ch_query = lambda dev: dev["max_output_channels"] == 4
    latence_metric = lambda dev: dev["default_high_output_latency"]

    devices = filter(name_query, devices)
    devices = filter(ch_query, devices)
    devices = sorted(devices, key=latence_metric)

    if len(devices):
        selected_device = devices[0]
        print(f"dualsence device count: {len(devices)}")
        print(selected_device)
        return selected_device["index"]

    return None

def find_normal_output():
    devices = sd.query_devices()
    name_query = lambda dev: dev["name"].find("404") > -1
    ch_query = lambda dev: dev["max_output_channels"] == 4
    latence_metric = lambda dev: dev["default_high_output_latency"]

    devices = filter(name_query, devices)
    devices = filter(ch_query, devices)
    devices = sorted(devices, key=latence_metric)

    if len(devices):
        return devices[0]["index"]

    return None


def simple_sine_gen(dev) -> sd.OutputStream:
        global hpc
        s_dev = sd.query_devices(dev, 'output')
        print(s_dev)
        samplerate = s_dev['default_samplerate']
        ch = s_dev["max_output_channels"]
        
        hpc.set_fs(samplerate)
        phase_factor = (2 * np.pi)/ samplerate

        def callback(outdata, frames, time, status):
            global hpc

            amp = hpc.amplitude
            freqs = hpc.freq_target * np.ones(frames)
            offset = hpc.phase_offset

            x = phase_factor * freqs
            phase_deltas = np.cumulative_sum(x)
            phase_array_0 = phase_deltas + hpc.signal_0_phase
            phase_array_1 = phase_deltas + hpc.signal_1_phase

            s1 = amp * np.sin(phase_array_0)
            s2 = amp * np.sin(phase_array_1 + offset)

            s1 = np.expand_dims(s1, axis=-1)
            s2 = np.expand_dims(s2, axis=-1)
        
            stereo_signal = np.concatenate((s1,s2), axis=-1)

            haptic_channel = 2
            outdata[:, haptic_channel:haptic_channel + 2] = stereo_signal
            hpc.signal_0_phase = phase_array_0[-1]
            hpc.signal_1_phase = phase_array_1[-1]

        stream = sd.OutputStream(device=dev, channels=ch, callback=callback,
                            samplerate=samplerate)
        
        return stream

def main():
    dev_idx = find_dualsence_audio_output()
    # dev_idx = find_normal_output()
    if dev_idx is None:
        print("!!! dualsence not connected by usb")
        exit(1)

    print(f"+++ find dualsence output devide with idx {dev_idx}")
    haptic_audio_stream = simple_sine_gen(dev_idx)
    inputs_loop(haptic_audio_stream)
    # dsloop(haptic_audio_stream)
    haptic_audio_stream.close()

main()

import matplotlib.pyplot as plt


def simple_graph():
    fs = 300
    n = fs # one sec
    initial_phase = np.pi / 6
    phase_factor = (2 * np.pi)/ fs
    freq = 2
    
    start_freq = 2
    end_freq = 20
    freq_dynamic = np.linspace(start_freq,end_freq,n)

    x = phase_factor * freq_dynamic

    y = np.cumulative_sum(x) + initial_phase
    z = np.sin(y)

    plt.plot(x)
    plt.plot(y)
    plt.plot(z)
    plt.show()


# simple_graph()
