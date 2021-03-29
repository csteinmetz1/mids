import os
import sys
import glob
import time
import resampy
import librosa
import numpy as np
import scipy.signal
from tqdm import tqdm
import soundfile as sf
import multiprocessing
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max

def fingerprint(
        audio_file : str,
        sample_rate : int = 22050,
        lowpass_fc : int = None,
        n_fft : int = 2048, 
        hop_length : int = 512,
        win_length : int = None,
        window : str = 'hann',
        use_log : bool = True,
        use_mel : bool = False,
        n_bins : int = 128,
        peak_distance : int = 18,
        target_zone_freq : int = 64,
        target_zone_time : int = 128,
        eps : float = 1e-16,
        plot : bool = False,
        **kwargs,
    ) -> list:
    """ Compute the fingerprint for an audio signal.
    
    Args:
        audio_file (str): Path to audio file to fingerprint.
        sample_rate (float): Desired sample rate of the audio signal.
        lowpass_fc (float): Lowpass pre-filter cutoff frequency.
        n_fft (int): Size of the FFT used for computation of the STFT.
        hop_length (int): Number of steps between each frame in the STFT.
        win_length (int): Size of the window. Defaults to `n_fft`.
        window (str): Window type to use. 
        use_log (bool): Represent STFT magnitudes on log-scale. 
        use_mel (bool): Project STFT bins in Mel-scale. 
        n_bins (int): Number of mel-frequency bins.
        peak_distance (int): Minimum distance between spectral peaks.
        target_zone_freq (int): Number of frequency bins above and below anchor.
        target_zone_time (int): Number of timesteps in front of anchor. 
        eps (float): Small epsilon for numerical stability. 
        plot (bool): Save a plot of the spectrogram and peaks.
    
    Returns:
        hasesh (list): List of hashes corresponding to the fingerprint.

    Note: If the audio loaded is at a different sample rate to the one specificed
    the audio will be resampled to match the specified sample rate. 
    """

    # load audio file
    x, sr = sf.read(audio_file)

    # resample if needed
    if sr != sample_rate:
        x = resampy.resample(x, sr, sample_rate) # resample to 22.05 kHz

    # peak normalize x
    x /= np.max(np.abs(x))

    # apply a lowpass filter
    if lowpass_fc is not None:
        sos = scipy.signal.butter(8, lowpass_fc, 'lp', fs=sample_rate, output='sos')
        x = scipy.signal.sosfilt(sos, x)

    # compute the STFT with the provided params
    X = librosa.stft(
        x, 
        n_fft=n_fft, 
        hop_length=hop_length,
        win_length=win_length,
        window=window
    )

    # magnitude
    X = np.abs(X)

    # apply a mel filterbank
    if use_mel:
        fb = librosa.filters.mel(sample_rate, n_fft, n_mels=n_bins)
        X = np.matmul(fb, X)

    # if we used a lowpass do not consider HF peaks
    if lowpass_fc is not None:
        bin_width = (1/2) * (sample_rate/n_fft)
        max_freq_bin = int(lowpass_fc / bin_width)
        X = X[:max_freq_bin,:]

    # convert to magnitude dB scale
    if use_log:
       X = 20 * np.log10(X + eps)

    # find the peaks
    peaks = peak_local_max(X, min_distance=peak_distance)
    
    filename = os.path.basename(audio_file).replace(".wav", "")
    if plot:
        plt.pcolormesh(X)
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

def find_matches(
        query_hashes : list, 
        db_songs : list,
        num_threads : int = 0,
    ) -> list:
    """ Find matches of a query with the database hashes.

    Args:
        query_hashes (list): list of pre-computed query fingerprint hashes. 
        db_songs (list): List of songs with their pre-computed hashes.
        num_threads (int): Number of parallel tasks to launch.

    Returns:
        matches (list): List of tuples that contain the song id, and
                        the score for each song in the database. 

    """

    matches = {}
    # iterate over each song in the database
    for db_song in db_songs:
        song_id = db_song["song_id"]
        inverted_lists = db_song["inverted_lists"]
        match_timesteps = {}
        for q_hash in query_hashes: # each hash is a query
            if q_hash["hash"] in inverted_lists: # check if hash is in inverted lists
                for timestep in inverted_lists[q_hash["hash"]]:
                    shifted = timestep - q_hash["timestep"]
                    if shifted not in match_timesteps:
                        match_timesteps[shifted] = 1
                    else:
                        match_timesteps[shifted] += 1

        if len(match_timesteps.values()) > 0:
            # return the max value of the mathcing function histogram
            matches[song_id] = max(match_timesteps.values())

    return matches

def compute_accuracy_metrics(
        query_files, 
        query_matches, 
        N=3
    ) -> dict:
    """ Compute the top-1 down to top-N acccuracy metrics. 

    Args:
        query_files (list): List of query filenames.
        query_matches (list): List of list of tuples containing (song_id, score).
        N (int): The lowest top-N match accuracy to compute. 

    Returns: 
        metrics (dict): Computed metrics.

    ex: 
    ```
    metrics = {
        "top-1" : 0.62,
        "top-2" : 0.71.
        ....
        "top-N" : 0.89,
    }
    ```
    """

    # check if we have same number of matches as queries
    assert(len(query_files) == len(query_matches))

    # total number of queries
    M = len(query_files)

    # storage for our metrics
    metrics = {}

    for query_file, matches in zip(query_files, query_matches):

        # get the ground truth song id
        query_id = os.path.basename(query_file).strip(".wav")
        song_id = query_id.split('-')[0]

        # sort the matches so that highest scoring are ranked first
        matches = [(db_song_id, score) for db_song_id, score in matches.items()]
        matches = sorted(matches, key=lambda a: a[1], reverse=True)

        for n in range(N):
            key = f"Top-{n+1}"
            # create a list containing just top-N matched song ids
            top_n_matches = [match[0] for match in matches[:n+1]]
            if song_id in top_n_matches:
                if key not in metrics:
                    metrics[key] = 1
                else:
                    metrics[key] += 1

    for key, val in metrics.items():
        metrics[key] /= M

    return metrics

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
    query_matches = [] # we will stop all matches here

    start = time.perf_counter()
    for query_file in query_files:
        query_hashes = fingerprint(query_file)
        query_matches.append(find_matches(query_hashes, db_songs))
    stop = time.perf_counter()
        
    print("-" * 64)
    metrics = compute_accuracy_metrics(query_files, query_matches)
    for key, val in metrics.items():
        print(f"{key} acc.  {val*100:0.2f}%")

    print()
    print(f"Analyized {len(query_files)} items in {stop-start:0.2f} s ({len(query_files)/(stop-start):0.1f} items/s)")
    print("-" * 64)
