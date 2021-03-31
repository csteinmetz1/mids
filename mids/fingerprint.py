import os
import sys
import glob
import resampy
import librosa
import numpy as np
import scipy.signal
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max

# optimal
# n_fft = 1024
# hop_length = 256
# peak_distance = 14
# target_zone_freq = 128
# target_zone_time = 256

def fingerprint(
        audio_file : str,
        sample_rate : int = 22050,
        lowpass_fc : int = 6000,
        n_fft : int = 2048, 
        hop_length : int = 512,
        win_length : int = None,
        window : str = 'hann',
        use_log : bool = True,
        use_mel : bool = False,
        n_bins : int = 128,
        peak_distance : int = 18,
        threshold_abs : float = -60,
        target_zone_freq : int = 256,
        target_zone_time : int = 128,
        max_peaks : int = 24,
        eps : float = 1e-16,
        plot : bool = True,
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
        threshold_abs (float): Minimum peak ampltiude in dB.
        target_zone_freq (int): Number of frequency bins above and below anchor.
        target_zone_time (int): Number of timesteps in front of anchor. 
        max_peaks (int): Maximum number of peaks within the target zone to consider.
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
    D = librosa.stft(
        x, 
        n_fft=n_fft, 
        hop_length=hop_length,
        win_length=win_length,
        window=window
    )

    # magnitude
    X = np.abs(D)
    X /= np.max(X)

    # apply a mel filterbank
    if use_mel:
        fb = librosa.filters.mel(sample_rate, n_fft, n_mels=n_bins)
        X = np.matmul(fb, X)

    # if we used a lowpass do not consider HF peaks
    bin_width = (1/2) * (sample_rate/n_fft)
    if lowpass_fc is not None:
        max_freq_bin = int(lowpass_fc / bin_width)
        X = X[:max_freq_bin,:]

    # convert to magnitude dB scale
    if use_log:
       X = 20 * np.log10(X + eps)
    
    peaks = peak_local_max(
                    X, 
                    min_distance=peak_distance, 
                    threshold_abs=threshold_abs
                )

    # peaks per second
    #print(len(peaks)/(x.shape[-1]/sample_rate))
    
    # optionally save out a plot of the spectrogram and peaks
    filename = os.path.basename(audio_file).replace(".wav", "")
    if plot:
        #D = librosa.amplitude_to_db(D, ref=np.max)
        #librosa.display.specshow(
        #                    D, 
        #                    sr=sample_rate, 
        #                    hop_length=hop_length, 
        #                    x_axis='time', 
        #                    y_axis='log')
        #plt.colorbar(format='%+2.0f dB')
        plt.scatter((peaks[:,1]*(n_fft/2))/sample_rate, peaks[:,0] * bin_width, facecolors='none', edgecolors='k')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.savefig(f'data/outputs/{filename}.png')
        plt.savefig(f'data/outputs/{filename}.pdf')
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
            if peaks_in_target_zone > max_peaks: break

    if len(hashes) < 1:
        print(f"No hashes found. {filename}")

    return hashes
