import os
import ffmpeg
import time
import requests
import gradio as gr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from pydub import AudioSegment
from .compare_speaker import CompareSpeaker, diarize

load_dotenv()

class GradioInference():
    def __init__(self):
        self.ref_audio_files = []

    def upload_files(self, files):
        self.ref_audio_files = [file.name for file in files]
        return self.ref_audio_files

    def update_spkr_ver_plot(self, res_df):
        sns.set_theme(style="dark")
        # sns.set_style(rc = {'figure.facecolor': '#94ADD7'})
        
        fig, ax = plt.subplots()
        chart = sns.lineplot(data=res_df, ax=ax, x='start', y='score')
        
        chart.set_title("Speaker Verification Scores", fontdict={'size': 12, 'color': 'grey'})
        chart.set_ylabel("Score", fontdict={'size': 10, 'color': 'grey'})
        chart.set_xlabel("Time", fontdict={'size': 10, 'color': 'grey'})
        chart.set_ylim(0, 1.0)


        freq = res_df.shape[0] // 10
        freq = 1 if freq == 0 else freq            
        # set the xlabels as the datetime data for the given labelling frequency,
        # also use only the date for the label
        ax.set_xticklabels(res_df.iloc[::freq]["start"])
        # set the xticks at the same frequency as the xlabels
        xtix = ax.get_xticks()
        ax.set_xticks(xtix[::freq])

        chart.tick_params(labelsize=8, color='grey')
        chart.get_figure().autofmt_xdate()
                
        return fig


    def __call__(self, video, window_res, mean_max_option):
        # Extract audio from video
        start_time = time.time()
        filename = os.path.basename(video)
        file_ext = filename.split('.')[-1]
        audio_segment_vid = AudioSegment.from_file(video, file_ext)
        audio_segment_vid.export(os.getenv("TMP_AUDIO_FILE"), format="wav")

        diarization_res = diarize(os.getenv("TMP_AUDIO_FILE"))
        cp_spkr = CompareSpeaker()
        spkr_ver_txt, spkr_ver_res_df = cp_spkr.iterate_timestamps(self.ref_audio_files, os.getenv("TMP_AUDIO_FILE"), diarization_res, resolution=window_res, method=mean_max_option)

        spkr_ver_plot = self.update_spkr_ver_plot(spkr_ver_res_df)
        
        end_time = time.time()
        time_taken = end_time-start_time
        

        return spkr_ver_txt, spkr_ver_plot, f"{round(time_taken, 2)}s taken"

gio = GradioInference()
examples_folder = os.getenv("EXAMPLES_FOLDER")
reference_folder = os.getenv("REFERENCE_FOLDER")

title = """
    <div style="text-align: center; max-width: 500px; margin: 0 auto;">
        <div>
        <h1>Speaker Verification</h1>
        </div>
    </div>
"""

with gr.Blocks() as demo:
    gr.HTML(title)

    with gr.Row().style(equal_height=True):
        with gr.Column(scale=0.5):
            video = gr.Video(label='Original Video')
            # File Upload for reference audio
            file_output = gr.File(label='Reference Audio')
            upload_button = gr.UploadButton("Click to Upload Reference Audio Files", file_types=["audio"], file_count="multiple")
            upload_button.upload(gio.upload_files, upload_button, file_output)

            window_res = gr.Slider(minimum=3, maximum=10, value=5, step=1, interactive=True, label="Window Size", info="Choose between 3 and 10s")
            mean_max_option = gr.Radio(
                ["mean", "max"], label="Method of comparison against ref audio", value='mean'
            )

        with gr.Column():
            plot = gr.Plot(label='Speaker Verification Scores')
            spkr_ver_txt = gr.Textbox(label='Speaker Verification Results', max_lines=5, show_copy_button=True, interactive=False)
            time_taken = gr.Markdown()
            btn = gr.Button("Leggo")       
        
    btn.click(gio, inputs=[video, window_res, mean_max_option], outputs=[spkr_ver_txt, plot, time_taken])
    
demo.queue(concurrency_count=3)
demo.launch()