import os
import torchaudio
import torch
from tqdm import tqdm
import argparse

def calculate_rms(tensor):
    return torch.sqrt(torch.mean(tensor**2))

def filter_audio_files(input_folder, output_folder, volume_threshold):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith('Noise.wav'):
            filepath = os.path.join(input_folder, filename)
            waveform, sample_rate = torchaudio.load(filepath)

            # Calculate RMS (Root Mean Square) volume
            rms_volume = calculate_rms(waveform)

            if rms_volume >= volume_threshold:
                output_path = os.path.join(output_folder, filename)
                torchaudio.save(output_path, waveform, sample_rate)

def main():
    parser = argparse.ArgumentParser(description="Filter audio files based on RMS volume.")
    parser.add_argument('input_folder', type=str, help='Path to the input folder containing audio files.')
    parser.add_argument('output_folder', type=str, help='Path to the output folder to save filtered audio files.')
    parser.add_argument('volume_threshold', type=float, help='Volume threshold for filtering audio files.', default=0.01)
    
    args = parser.parse_args()

    filter_audio_files(args.input_folder, args.output_folder, args.volume_threshold)

if __name__ == "__main__":
    main()
