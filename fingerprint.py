import os
import sys
import glob
import torch
import julius
import librosa
import torchaudio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max

def fingerprint(
    x : torch.Tensor, 
    sample_rate : int,
    n_fft : int = 2048, 
    hop_length : int = 1024,
    n_bins : int = 128,
    filename : str = None, 
    ) -> torch.Tensor:
    """ Compute the finger print for an audio signal.
    
    Args:
        x (tensor): A 1d tensor containing the audio samples.
        sample_rate (float): Sample rate of the audio signal.
        n_fft (int): Size of the FFT used for computation of the STFT.
        hop_length (int): Number of steps between each frame in the STFT.
        n_bins (int): Number of mel-frequency bins
    
    Returns:
        f (tensor): Fingerprint corresponding to the input audio signal.
    """

    # peak normalize x
    x /= x.abs().max()

    X = torch.stft(
        x, 
        n_fft, 
        hop_length=hop_length, 
        return_complex=True)

    X_mag = X.abs().squeeze()

    # apply a mel filterbank
    fb = librosa.filters.mel(sample_rate, n_fft, n_mels=n_bins)
    fb = torch.tensor(fb)
    X_mag = torch.matmul(fb, X_mag)

    X_mag_dB = 20 * torch.log(X_mag.clamp(1e-8))

    peaks = peak_local_max(X_mag_dB.numpy(), min_distance=10)

    if filename is not None:
        plt.pcolormesh(X_mag_dB)
        plt.scatter(peaks[:,1], peaks[:,0], facecolors='none', edgecolors='k')
        plt.savefig(f'data/outputs/{filename}_fingerprint.png')
        plt.close('all')

    # what is the best way to define the target zone?
    target_zone_freq = 24
    target_zone_time = 50

    # sort the peaks so they are aligned by time axis
    peaks = peaks[np.argsort(peaks[:, 1])]

    hashes = []
    # iterate over each peak
    for n in range(peaks.shape[0]):
        anchor = peaks[n,:]

        # iterate over all peaks again to 
        # find those peaks within target zone
        for i in range(peaks.shape[0]):
            point = peaks[i,:]
            if ((point[0] < anchor[0] + target_zone_freq) and (point[0] > anchor[0] - target_zone_freq) and 
                (point[1] < anchor[1] + target_zone_time) and (point[1] > anchor[1] + 5) and 
                not np.array_equal(point, anchor)):
                hashes.append({
                    "hash" : (anchor[0], point[0], point[1] - anchor[1]),
                    "song" : filename,
                    "timestep" : anchor[1]
                })
                # generate a hash (time index, (anchor freq, point freq, time delta))

    return hashes

if __name__ == '__main__':

    database_dir = 'data/database_recordings/'
    query_dir = 'data/query_recordings/'

    # generate hashes for all items in database
    db_hashes = []
    database_files = sorted(glob.glob(os.path.join(database_dir, "*.wav")))
    print("Building database...")
    for database_file in tqdm(database_files, ncols=80):
        x_db, sr_db = torchaudio.load(database_file)
        db_hashes += fingerprint(x_db, sr_db, filename=os.path.basename(database_file))

    query_files = sorted(glob.glob(os.path.join(query_dir, "*.wav")))
    for query_file in query_files:
        print("-" * 64)
        x_q, sr_q = torchaudio.load(query_file)
        x_q = julius.resample_frac(x_q, sr_q, sr_db) # resample to 22.05 kHz
        q_hashes = fingerprint(x_q, sr_db, filename=os.path.basename(query_file))

        matches = {}

        for q_hash in q_hashes:
            for db_hash in db_hashes:
                if db_hash["hash"] == q_hash["hash"]:
                    if db_hash["song"] not in matches:
                        matches[db_hash["song"]] = 1
                    else:
                        matches[db_hash["song"]] += 1

        matches = dict(sorted(matches.items(), key=lambda item: item[1], reverse=True))
        print(f"query: {os.path.basename(query_file).strip('.wav')}")
        for idx, (song, count) in enumerate(matches.items()):
            print(idx+1, song, count)
            if idx+1 > 2: break

