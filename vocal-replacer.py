from audio_separator.separator import Separator
import argparse
from yt_dlp import YoutubeDL
from pydub import AudioSegment
from pathlib import Path
import os

from dotenv import load_dotenv
from scipy.io import wavfile

from rvc.modules.vc.modules import VC

load_dotenv(".env")

vc = VC()

separator = Separator()

# Load a model
separator.load_model(model_filename='vocals_mel_band_roformer.ckpt')
output_names = {
    "Vocals": "vocals_output",
    "Instrumental": "instrumental_output",
    "Other": "instrumental_output"
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

def replace_vocals(url: str, name: str, pitch: int):
    model = f"./models/{name}/model.pth"
    model_index = f"./models/{name}/model.index"
    vc.get_vc(model)
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        info_with_audio_extension = dict(info)
        info_with_audio_extension['ext'] = 'wav'
        final_filename = ydl.prepare_filename(info_with_audio_extension)

    output_files = separator.separate(final_filename, output_names)

    tgt_sr, audio_opt, times, _ = vc.vc_single(
                1,
                Path("vocals_output.wav"),
                pitch,
                'rmvpe',
                index_file=model_index,
                filter_radius=10,
                protect=0,
                index_rate=0.33
        )
    wavfile.write("vocals_output.wav", tgt_sr, audio_opt)

    sound1 = AudioSegment.from_file("vocals_output.wav")
    sound2 = AudioSegment.from_file("instrumental_output.wav")

    combined = sound1.overlay(sound2)

    os.unlink(final_filename)
    os.unlink("vocals_output.wav")
    os.unlink("instrumental_output.wav")

    exported_name = f"output/{name}_" + final_filename.replace('.wav', '.mp3')

    combined.export(exported_name, format='mp3', bitrate="320k")
    with open(exported_name, "rb") as file:
        return file.read()
    
if __name__ == "__main__":
    #args
    parser = argparse.ArgumentParser(
                        prog='Vocal Replacer',
                        description='Replaces vocals from a youtube video',
                        epilog='By: Alexankitty')

    parser.add_argument("voice")
    parser.add_argument("url")

    parser.add_argument("-p", "--pitch", default=0, type=int, help="Transpose (integer, number of semitones)")
    args = parser.parse_args()

    replace_vocals(args.url, args.voice, args.pitch )