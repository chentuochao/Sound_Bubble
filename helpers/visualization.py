import os

import numpy as np

import sep.helpers.utils as utils

import subprocess
import json

# import required module
from playsound import playsound


def play_audio(fname):
    # for playing note.wav file
    playsound(fname)
    # print('playing sound using  playsound')
    # subprocess.Popen(['play', fname],stdout=subprocess.DEVNULL)

def onclick(event, input_dir, metadata, query_positions, writing_dir):
    if event.dblclick:
        if event.button == 1:
            query_positions = np.array(query_positions)
            pos = np.array([event.xdata, event.ydata])
            i = np.argmin(np.linalg.norm(query_positions - pos, axis=1))
            if(np.linalg.norm(query_positions[i] - pos) < 0.5):
                fname = f"output_{i}.wav"
                fname = (os.path.join(writing_dir, fname))
                play_audio(fname)
            else:
                voices = [(x, metadata[x]) for x in metadata if 'voice' in x]
                voice_positions = np.array([x[1]['position'] for x in voices])
                i = np.argmin(np.linalg.norm(voice_positions - pos, axis=1))
                if (np.linalg.norm(voice_positions[i] - pos) < 0.5):
                    fname = 'mic00_' + voices[i][0] + '.wav'
                    fname = (os.path.join(input_dir, fname))
                    play_audio(fname)


def draw_diagram(input_dir, query_positions, query_radius, writing_dir):
    """
    Draws the setup of all the voices in space, and colored triangles for the beams
    """
    import matplotlib
    matplotlib.use("tkAgg")
    import matplotlib.pyplot as plt
    matplotlib.style.use('ggplot')

    with open(os.path.join(input_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    mic_positions = np.array([metadata[x]['position'] for x in metadata if 'mic' in x])[:, :2]
    voice_positions = np.array([metadata[x]['position'] for x in metadata if 'voice' in x])[:, :2]
    query_positions = query_positions[:, :2]

    fig, ax = plt.subplots()
    ax.set(xlim=(-5, 5), ylim = (-5, 5))
    ax.set_aspect("equal")

    print(voice_positions)
    print(mic_positions)

    plt.tick_params(axis='both',
        which='both', bottom='off',
        top='off', labelbottom='off', right='off', left='off', labelleft='off'
    )

    for idx, query_position in enumerate(query_positions):
        query_position = tuple(query_position)
        circle = plt.Circle(query_position, radius=0.1, color='green')
        ax.add_patch(circle)
        circle = plt.Circle(query_position, radius=query_radius, fill=False, color='green')
        ax.add_patch(circle)

    for idx, voice_position in enumerate(voice_positions):
        voice_position = tuple(voice_position)
        circle = plt.Circle(voice_position, radius=0.12, color='red')
        ax.add_patch(circle)
        
    for idx, mic_position in enumerate(mic_positions):
        mic_position = tuple(mic_position)
        circle = plt.Circle(mic_position, radius=0.013, color='blue')
        ax.add_patch(circle)

    # ax.set_xlim([-20,20])
    # ax.set_ylim([-20,20])
    ax.tick_params(axis='both', which='both', labelcolor="white", colors="white")
    fig.canvas.mpl_connect('button_press_event', lambda x: onclick(x, input_dir, metadata, query_positions, writing_dir))
    plt.show()