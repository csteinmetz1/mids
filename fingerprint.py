import os
import sys
import glob
import time
import torch
import julius
import librosa
import torchaudio
import numpy as np
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max

def fingerprint(
    audio_file : str,
    sample_rate : int = 22050,
    n_fft : int = 2048, 
    hop_length : int = 1024,
    n_bins : int = 128,
    plot: bool = False,
    ) -> torch.Tensor:
    """ Compute the finger print for an audio signal.
    
    Args:
        audio_file (str): Path to audio file to fingerprint.
        sample_rate (float): Desired sample rate of the audio signal.
        n_fft (int): Size of the FFT used for computation of the STFT.
        hop_length (int): Number of steps between each frame in the STFT.
        n_bins (int): Number of mel-frequency bins.
        plot (bool): Save a plot of the spectrogram and peaks.
    
    Returns:
        f (tensor): Fingerprint corresponding to the input audio signal.

    Note: If the audio loaded is at a different sample rate to the one specificed
    the audio will be resampled to match the specified sample rate. 
    """

    # load audio file
    x, sr = torchaudio.load(audio_file)

    # resample if needed
    if sr != sample_rate:
        x = julius.resample_frac(x, sr, sample_rate) # resample to 22.05 kHz

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
    
    filename = os.path.basename(audio_file).replace(".wav", "")
    if plot:
        plt.pcolormesh(X_mag_dB)
        plt.scatter(peaks[:,1], peaks[:,0], facecolors='none', edgecolors='k')
        plt.savefig(f'data/outputs/{filename}.png')
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

def find_matches(query_file, db_hashes):
    q_hashes = fingerprint(query_file)

    matches = {}
    for q_hash in q_hashes:
        for db_hash in db_hashes:
            if db_hash["hash"] == q_hash["hash"]:
                if db_hash["song"] not in matches:
                    matches[db_hash["song"]] = 1
                else:
                    matches[db_hash["song"]] += 1

    matches = dict(sorted(matches.items(), key=lambda item: item[1], reverse=True))
    #print(f"query: {os.path.basename(query_file).strip('.wav')}")
    top_matches = []
    for idx, (song, count) in enumerate(matches.items()):
        top_matches.append(song)
        if idx+1 > 2: break

    return top_matches

if __name__ == '__main__':

    database_dir = 'data/database_recordings/'
    query_dir = 'data/query_recordings/'

    # generate hashes for all items in database
    database_files = sorted(glob.glob(os.path.join(database_dir, "*.wav")))
    print("Building database...")
    start = time.perf_counter()
    with multiprocessing.Pool(processes=12) as pool:
        results = pool.map(fingerprint, database_files)
    stop = time.perf_counter()
    print(f"Built database of {len(database_files)} items in {stop-start:0.2f} s ({len(database_files)/stop-start:0.1f} items/s)")

    db_hashes = []
    for db_hash_list in results:
        db_hashes += db_hash_list

    query_files = sorted(glob.glob(os.path.join(query_dir, "*.wav")))
    params = zip(query_files, [db_hashes] * len(query_files))
    start = time.perf_counter()
    with multiprocessing.Pool(processes=12) as pool:
        results = pool.starmap(find_matches, params)
    stop = time.perf_counter()
    print(f"Analyized {len(query_files)} items in {stop-start:0.2f} s ({len(query_files)/stop-start:0.1f} items/s)")

    for query_file, query_result in zip(query_files, results):
        print(os.path.basename(query_file), query_result)

