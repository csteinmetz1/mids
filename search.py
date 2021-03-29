import os
import glob
import time
import pickle
import multiprocessing
import matplotlib.pyplot as plt

from tqdm import tqdm
from mids.fingerprint import fingerprint, find_matches, compute_accuracy_metrics

def fingerprint_wrapper(kwargs):
    return fingerprint(**kwargs)

def find_matches_wrapper(kwargs):
    return find_matches(**kwargs)

def search(
        database_dir : str, 
        query_dir : str,
        params : dict,
        num_threads : int = 0,
    ) -> dict:
    """ Compute database hashes and test accuracy on query set.

    Args:
        database_dir (str): Path to the database audio files.
        query_dir (str): Path to the query audio files.
        params (dict): Dict containing the fingerprinting parameters.
        num_threads (int): Number of parallel threads for databaase generation.

    Returns:
        search_results (dict): Dict containing the accuracy and timing. 
    """

    # generate hashes for all items in database
    database_files = sorted(glob.glob(os.path.join(database_dir, "*.wav")))
    db_songs_hashes = []

    map_params = []
    for database_file in database_files:
        database_params = params.copy()
        database_params["audio_file"] = database_file
        map_params.append(database_params)

    print()
    print("Generating hashes from database...")
    start = time.perf_counter()
    with multiprocessing.Pool(processes=24) as pool:
        db_songs_hashes = pool.map(fingerprint_wrapper, map_params)
    stop = time.perf_counter()
    database_time = stop-start

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

    # setup the map parameters 
    map_params = []
    for query_file in query_files:
        query_params = params.copy()
        query_params["audio_file"] = query_file
        map_params.append(query_params)

    print("Generating hashes from queries...")
    start = time.perf_counter()
    with multiprocessing.Pool(processes=24) as pool:
        query_songs_hashes = pool.map(fingerprint_wrapper, map_params)

    query_matches = []
    print("Finding matches...")
    for query_hashes in query_songs_hashes:
        query_matches.append(find_matches(query_hashes, db_songs))
    stop = time.perf_counter()

    #query_files = sorted(glob.glob(os.path.join(query_dir, "*.wav")))
    #query_matches = [] # we will stop all matches here

    #start = time.perf_counter()
    #for query_file in query_files:
    #    query_hashes = fingerprint(query_file)
    #    query_matches.append(find_matches(query_hashes, db_songs))
    #stop = time.perf_counter()

    query_time = stop-start
        
    metrics = compute_accuracy_metrics(query_files, query_matches)

    search_result = {
        "params" : params,
        "metrics" : metrics,
        "database_time" : database_time,
        "query_time" : query_time
    }

    return search_result

if __name__ == '__main__':

    database_dir = 'data/database_recordings/'
    query_dir = 'data/query_recordings/'

    # minium peak distances for peak picking
    #peak_distances = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36]
    #n_ffts = [128, 256, 512, 1024, 2048, 4096, 8192]
    #hop_lengths = [int(n_fft/2) for n_fft in n_ffts]
    #hop_lengths = [1/12, 1/10, 1/8, 1/6, 1/4, 1/3, 1/2]
    #target_zone_freqs = [16, 24, 32, 48, 64, 128, 256, 512]
    #target_zone_times = [16, 24, 32, 48, 64, 128, 256, 512]
    lowpass_fcs = [1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000]

    search_results = []
    param_name = "lowpass_fc"

    #for hop_length in tqdm(hop_lengths, ncols=80):
    #for n_fft, hop_length in tqdm(zip(n_ffts, hop_lengths), ncols=80):
    #for peak_distance in tqdm(peak_distances, ncols=80):
    #for target_zone_freq in tqdm(target_zone_freqs, ncols=80):
    #for target_zone_time in tqdm(target_zone_times, ncols=80):
    for lowpass_fc in tqdm(lowpass_fcs, ncols=80):

        params = {
            "n_fft" : 1024,
            "hop_length" : 256,
            "peak_distance" : 14,
            "target_zone_freq" : 128,
            "target_zone_time" : 256,
            "lowpass_fc" : 6000
        }

        search_results.append(
            search(
                database_dir,
                query_dir, 
                params
            ))

    with open(os.path.join("data", "search", f'{param_name}.pkl'), 'wb') as f:
        pickle.dump(search_results, f)


    