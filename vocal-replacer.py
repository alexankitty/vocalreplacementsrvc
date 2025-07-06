from rvc_python.infer import RVCInference
from audio_separator.separator import Separator
import argparse
from yt_dlp import YoutubeDL
from pydub import AudioSegment

#args
parser = argparse.ArgumentParser(
                    prog='Vocal Replacer',
                    description='Replaces vocals from a youtube video',
                    epilog='By: Alexankitty')

parser.add_argument("url")
args = parser.parse_args()

# rvc
rvc = RVCInference(device="cuda:0")
rvc.load_model("./models/miku_default_rvc/miku_default_rvc.pth", index_path="./models/miku_default_rvc/added_IVF4457_Flat_nprobe_1_miku_default_rvc_v2.index")
rvc.set_params(f0method="crepe")

# Initialize the Separator class (with optional configuration properties, below)
separator = Separator()

# Load a model
separator.load_model(model_filename='model_bs_roformer_ep_317_sdr_12.9755.ckpt')
output_names = {
    "Vocals": "vocals_output",
    "Instrumental": "instrumental_output",
}

ydl_opts = {
    'format': 'bestaudio/best',
    # ℹ️ See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
    'postprocessors': [{  # Extract audio using ffmpeg
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
    }],
    'outtmpl': '%(title)s.%(ext)s'
}

with YoutubeDL(ydl_opts) as ydl:
    info_dict = ydl.extract_info(args.url, download=False)
    audio_filename = info_dict.get('title', None) + '.wav'
    ydl.download([args.url])

# Separate all audio files located in a folder
output_files = separator.separate(audio_filename, output_names)

rvc.infer_file("vocals_output.wav", "vocals_output.wav")

sound1 = AudioSegment.from_file("vocals_output.wav")
sound2 = AudioSegment.from_file("instrumental_output.wav")

combined = sound1.overlay(sound2)

combined.export("output/Miku " + audio_filename, format='wav')
