import os
import glob
import resampy
import librosa
import numpy as np
import scipy.signal
import soundfile as sf
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max

def fingerprint(
        audio_file : str,
        sample_rate : int = 22050,
        lowpass_fc : int = 6000,
        n_fft : int = 1024, 
        hop_length : int = 256,
        win_length : int = None,
        window : str = 'hann',
        use_log : bool = True,
        use_mel : bool = False,
        n_bins : int = 128,
        peak_distance : int = 14,
        target_zone_freq : int = 128,
        target_zone_time : int = 256,
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
    
    # optionally save out a plot of the spectrogram and peaks
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
            point = peaks[i,:] # find peaks within the target zone to generate hashes
            if ((point[0] < anchor[0] + target_zone_freq) and (point[0] > anchor[0] - target_zone_freq) and 
                (point[1] < anchor[1] + target_zone_time) and (point[1] > anchor[1] + 1) and 
                not np.array_equal(point, anchor)):
                hashes.append({
                    "hash" : (anchor[0], point[0], point[1] - anchor[1]),
                            # (anchor freq, point freqs, time difference)
                    "song" : filename,
                    "timestep" : anchor[1]
                })
                peaks_in_target_zone += 1

    return hashes
