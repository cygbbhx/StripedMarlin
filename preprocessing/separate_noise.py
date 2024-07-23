import sys 
sys.path.append('preprocessing/vocal_remover')

import argparse
import os
from vocal_remover.inference import Separator
from vocal_remover.lib import nets, spec_utils
from tqdm import tqdm
import soundfile as sf
import torch
import librosa
import numpy as np

MODEL_DIR = os.path.join('preprocessing/vocal_remover', 'models')
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, 'baseline.pth')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--pretrained_model', '-P', type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--sr', '-r', type=int, default=32000)
    p.add_argument('--n_fft', '-f', type=int, default=2048)
    p.add_argument('--hop_length', '-H', type=int, default=1024)
    p.add_argument('--batchsize', '-B', type=int, default=4)
    p.add_argument('--cropsize', '-c', type=int, default=256)
    p.add_argument('--output_image', '-I', action='store_true')
    p.add_argument('--tta', '-t', action='store_true')
    p.add_argument('--postprocess', '-p', action='store_true')
    p.add_argument('--output_dir', '-o', type=str, default="")
    p.add_argument('--save_option', '-s', type=str, choices=['vocal-only', 'noise-only', 'both'], default='both')
    args = p.parse_args()

    # print('loading model...', end=' ')
    device = torch.device('cpu')
    if args.gpu >= 0:
        if torch.cuda.is_available():
            device = torch.device('cuda:{}'.format(args.gpu))
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device('mps')
    model = nets.CascadedNet(args.n_fft, args.hop_length, 32, 128)
    model.load_state_dict(torch.load(args.pretrained_model, map_location='cpu'))
    model.to(device)

    # data_root = '/home/work/StripedMarlin/contest_data/unlabeled_data'
    # args.output_dir = '/home/work/StripedMarlin/sohyun/StripedMarlin/processed_data'
    
    sp = Separator(
        model=model,
        device=device,
        batchsize=args.batchsize,
        cropsize=args.cropsize,
        postprocess=args.postprocess
    )

    for file in tqdm(os.listdir(args.input)):
        input_file = os.path.join(args.input, file)
        # args.input = os.path.join(data_root, file)
        X, sr = librosa.load(
            input_file, sr=args.sr, mono=False, dtype=np.float32, res_type='kaiser_fast'
        )
        basename = os.path.splitext(os.path.basename(input_file))[0]

        if X.ndim == 1:
            X = np.asarray([X, X])

        X_spec = spec_utils.wave_to_spectrogram(X, args.hop_length, args.n_fft)

        if args.tta:
            y_spec, v_spec = sp.separate_tta(X_spec)
        else:
            y_spec, v_spec = sp.separate(X_spec)

        output_dir = args.output_dir
        if output_dir != "":  # modifies output_dir if theres an arg specified
            output_dir = output_dir.rstrip('/') + '/'
            os.makedirs(output_dir, exist_ok=True)

        if args.save_option in ['both', 'noise-only']:
            wave = spec_utils.spectrogram_to_wave(y_spec, hop_length=args.hop_length)
            sf.write('{}{}_Noise.wav'.format(output_dir, basename), wave.T, sr)

        if args.save_option in ['both', 'vocal-only']:
            wave = spec_utils.spectrogram_to_wave(v_spec, hop_length=args.hop_length)
            wave = np.mean(wave, axis=0)
            sf.write('{}{}_Vocals.wav'.format(output_dir, basename), wave.T, sr)


if __name__ == '__main__':
    main()
