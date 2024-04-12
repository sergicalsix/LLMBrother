from moviepy.editor import *
from pydub import AudioSegment
import glob 

def convert_mkv_to_wav(input_path, output_path):
    video = AudioFileClip(input_path)
    video.write_audiofile(output_path)


def split_wav_by_duration(input_file, output_prefix, duration_ms):
    audio = AudioSegment.from_wav(input_file)
    for i, chunk in enumerate(audio[::duration_ms]):
        chunk.export(f"{output_prefix}_{i}.wav", format="wav")


if __name__ == "__main__":
    input_file_list = glob.glob("*.mkv")
    for input_path in input_file_list:
        wav_path = input_path.replace(".mkv", ".wav")
        convert_mkv_to_wav(input_path, wav_path)

        output_prefix = wav_path.replace(".wav", "")
        duration_ms = 60_000  # ミリ秒
        split_wav_by_duration(wav_path, output_prefix, duration_ms)

