import os
import sys
import glob
import time
import torch
import julius
import librosa
import torchaudio
import numpy as np
import scipy.signal
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max

def fingerprint(
    audio_file : str,
    sample_rate : int = 22050,
    n_fft : int = 2048, 
    hop_length : int = 512,
    use_log : bool = True,
    use_mel : bool = False,
    n_bins : int = 128,
    peak_distace : int = 16,
    target_zone_freq : int = 32,
    target_zone_time : int = 64,
    plot : bool = True,
    ) -> torch.Tensor:
    """ Compute the fingerprint for an audio signal.
    
    Args:
        audio_file (str): Path to audio file to fingerprint.
        sample_rate (float): Desired sample rate of the audio signal.
        n_fft (int): Size of the FFT used for computation of the STFT.
        hop_length (int): Number of steps between each frame in the STFT.
        use_log (bool): Represent STFT magnitudes on log-scale. 
        use_mel (bool): Project STFT bins in Mel-scale. 
        n_bins (int): Number of mel-frequency bins.
        peak_distance (int): Minimum distance between spectral peaks.
        target_zone_freq (int): Number of frequency bins above and below anchor.
        target_zone_time (int): Number of timesteps in front of anchor. 
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

    # apply a lowpass filter
    sos = scipy.signal.butter(8, 8000, 'lp', fs=sample_rate, output='sos')
    x = scipy.signal.sosfilt(sos, x.numpy())
    x = torch.tensor(x)

    X = torch.stft(
        x, 
        n_fft, 
        hop_length=hop_length, 
        return_complex=True)

    X_mag = X.abs().squeeze()

    # apply a mel filterbank
    if use_mel:
        fb = librosa.filters.mel(sample_rate, n_fft, n_mels=n_bins)
        fb = torch.tensor(fb)
        X_mag = torch.matmul(fb, X_mag)

    if use_log:
        X_mag = 20 * torch.log(X_mag.clamp(1e-8))

    peaks = peak_local_max(X_mag.numpy(), min_distance=peak_distace)
    
    filename = os.path.basename(audio_file).replace(".wav", "")
    if plot:
        plt.pcolormesh(X_mag)
        plt.scatter(peaks[:,1], peaks[:,0], facecolors='none', edgecolors='k')
        plt.savefig(f'data/outputs/{filename}.png')
        plt.close('all')

    # sort the peaks so they are aligned by time axis
    peaks = peaks[np.argsort(peaks[:, 1])]

    hashes = []
    # iterate over each peak
    for n in range(peaks.shape[0]):
        anchor = peaks[n,:]

        # iterate over all peaks again to 
        # find those peaks within target zone
        peaks_in_target_zone = 0
        for i in range(peaks.shape[0]):
            point = peaks[i,:]
            if ((point[0] < anchor[0] + target_zone_freq) and (point[0] > anchor[0] - target_zone_freq) and 
                (point[1] < anchor[1] + target_zone_time) and (point[1] > anchor[1] + 1) and 
                not np.array_equal(point, anchor)):
                hashes.append({
                    "hash" : (anchor[0], point[0], point[1] - anchor[1]),
                    "song" : filename,
                    "timestep" : anchor[1]
                })
                peaks_in_target_zone += 1

    return hashes

def find_matches(query_file, db_songs):
    q_hashes = fingerprint(query_file)

    matches = {}
    # iterate over each song in the database
    for db_song in db_songs:
        song_id = db_song["song_id"]
        inverted_lists = db_song["inverted_lists"]
        match_timesteps = {}
        # check each query hash
        for q_hash in q_hashes: # each hash is a query
            if q_hash["hash"] in inverted_lists: # check if hash is in inverted lists
                for timestep in inverted_lists[q_hash["hash"]]:
                    shifted = timestep - q_hash["timestep"]
                    if shifted not in match_timesteps:
                        match_timesteps[shifted] = 1
                    else:
                        match_timesteps[shifted] += 1
        if len(match_timesteps.values()) > 0:
            matches[song_id] = max(match_timesteps.values())
            #matches[song_id] = sum(match_timesteps.values())

    # sort the matches
    matches = [(song_id, score) for song_id, score in matches.items()]
    matches = sorted(matches, key=lambda a: a[1], reverse=True)

    return matches

if __name__ == '__main__':

    database_dir = 'data/database_recordings/'
    query_dir = 'data/query_recordings/'

    # generate hashes for all items in database
    database_files = sorted(glob.glob(os.path.join(database_dir, "*.wav")))
    print("Building database...")
    start = time.perf_counter()
    with multiprocessing.Pool(processes=12) as pool:
        db_songs_hashes = pool.map(fingerprint, database_files)
    stop = time.perf_counter()
    print(f"Built database of {len(database_files)} items in {stop-start:0.2f} s ({len(database_files)/(stop-start):0.1f} items/s)")

    db_songs = []

    for db_song in db_songs_hashes:
        db_song_inv_lists = {}
        song_id = db_song[0]["song"]
        for db_song_hash in db_song:
            if db_song_hash["hash"] not in db_song_inv_lists:
                db_song_inv_lists[db_song_hash["hash"]] = [db_song_hash["timestep"]]
            else:
                db_song_inv_lists[db_song_hash["hash"]] += db_song_hash["timestep"]
        db_songs.append({
            "song_id" : song_id,
            "inverted_lists" : db_song_inv_lists
        })

    query_files = sorted(glob.glob(os.path.join(query_dir, "*.wav")))

    # metrics 
    correct = 0
    incorrect = 0

    start = time.perf_counter()
    for query_file in query_files:
        matches = find_matches(query_file, db_songs)
        print(os.path.basename(query_file))
        print(matches[:3])
        print("-" * 64)
        gt_song_id = os.path.basename(query_file).split("-")[0]
        if gt_song_id == matches[0][0]:
            correct += 1
        else:
            incorrect += 1 

    stop = time.perf_counter()
    accuracy = correct / (correct + incorrect)
    print(f"Accuracy: {accuracy*100:0.2f}%")
    print(f"Analyized {len(query_files)} items in {stop-start:0.2f} s ({len(query_files)/(stop-start):0.1f} items/s)")
    print("-" * 64)
