from pydub import AudioSegment
import os
import random


def split_audio_random(
    input_path, output_folder, segment_index, min_length=180, max_length=300
):
    audio = AudioSegment.from_file(input_path, format="ogg")
    total_duration = len(audio) / 1000  # Convert to seconds

    start_time = 0
    while start_time < total_duration:
        segment_length = random.randint(min_length, max_length)

        if start_time + segment_length > total_duration:
            segment_length = total_duration - start_time

        end_time = start_time + segment_length

        segment = audio[int(start_time * 1000) : int(end_time * 1000)]

        output_path = os.path.join(output_folder, f"segment_{segment_index:03d}.wav")
        segment.export(output_path, format="wav")

        print(f"Exported: {output_path} ({segment_length:.2f}s)")

        start_time = end_time
        segment_index += 1
    return segment_index


def process_folder(input_folder, output_folder):
    audio_files = [f for f in os.listdir(input_folder) if f.endswith(".ogg")]
    index = 0
    for i, audio_file in enumerate(audio_files):
        # os.makedirs(output_folder, exist_ok=True)
        input_path = os.path.join(input_folder, audio_file)
        print(f"Processing: {audio_file}")
        index = split_audio_random(input_path, output_folder, index)
        print(
            f"Finished processing: {audio_file}\nLeft {len(audio_files) - i - 1} files"
        )


input_folder = "INPUT_FOLDER_PATH"
output_folder = "OUTPUT_FOLDER_PATH"
process_folder(input_folder, output_folder)
