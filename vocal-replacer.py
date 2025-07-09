from audio_separator.separator import Separator
import argparse
from yt_dlp import YoutubeDL
from pydub import AudioSegment
from pathlib import Path

from dotenv import load_dotenv
from scipy.io import wavfile

from rvc.modules.vc.modules import VC

load_dotenv(".env")



#args
parser = argparse.ArgumentParser(
                    prog='Vocal Replacer',
                    description='Replaces vocals from a youtube video',
                    epilog='By: Alexankitty')

parser.add_argument("voice")
parser.add_argument("url")

parser.add_argument("-p", "--pitch", default=0, type=int, help="Transpose (integer, number of semitones)")
args = parser.parse_args()

model_name = args.voice

model = f"./models/{model_name}/model.pth"
model_index = f"./models/{model_name}/model.index"

# rvc
#rvc = RVCInference(device="cuda:0")
#rvc.load_model("./models/miku_default_rvc/miku_default_rvc.pth", index_path="./models/miku_default_rvc/added_IVF4457_Flat_nprobe_1_miku_default_rvc_v2.index")
#rvc.set_params(f0method="crepe",filter_radius=100, protect=1, index_rate=1, f0up_key=args.pitch)

vc = VC()
vc.get_vc(model)


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
    'outtmpl': '%(title)s.%(ext)s',
    'restrictfilenames': True
}

with YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(args.url, download=True)
    info_with_audio_extension = dict(info)
    info_with_audio_extension['ext'] = 'wav'
    final_filename = ydl.prepare_filename(info_with_audio_extension)


# Separate all audio files located in a folder
output_files = separator.separate(final_filename, output_names)

#rvc.infer_file("vocals_output.wav", "vocals_output.wav")


tgt_sr, audio_opt, times, _ = vc.vc_single(
            1,
            Path("vocals_output.wav"),
            args.pitch,
            'rmvpe',
            index_file=model_index,
            filter_radius=10,
            protect=0
      )
wavfile.write("vocals_output.wav", tgt_sr, audio_opt)

sound1 = AudioSegment.from_file("vocals_output.wav")
sound2 = AudioSegment.from_file("instrumental_output.wav")

combined = sound1.overlay(sound2)

combined.export(f"output/{model_name}_" + final_filename, format='wav')
