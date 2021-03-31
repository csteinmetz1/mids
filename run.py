import os
import sys
import time
import glob
import pickle
import numpy as np
import multiprocessing

from mids.fingerprint import fingerprint
from mids.match import find_matches, compute_accuracy_metrics

def fingerprintBuilder(
        database_dir : str,
        fingerprints_filepath : str = "fingerprints.pkl",
        num_threads : int = 0,
    ):
    
    # generate hashes for all items in database
    database_files = sorted(glob.glob(os.path.join(database_dir, "*.wav")))
    print("Building database...")
    start = time.perf_counter()

    if num_threads > 0: # use multithreading
        with multiprocessing.Pool(processes=num_threads) as pool:
            db_songs_hashes = pool.map(fingerprint, database_files)
    else:
        db_songs_hashes = []
        for database_file in database_files:
            db_songs_hashes.append(fingerprint(database_file))

    stop = time.perf_counter()
    print(f"Built database of {len(database_files)} items in {stop-start:0.2f} s ({len(database_files)/(stop-start):0.1f} items/s)")

    db_songs = []

    # structure the list of songs
    # create a list of dicts, which the song id, and inverted lists
    for db_song in db_songs_hashes:
        db_song_inv_lists = {}
        if len(db_song) < 1:
            print(db_song)
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

    # save list of database song hashes to disk
    with open(fingerprints_filepath, 'wb') as f:
        pickle.dump(db_songs, f)

def audioIdentification(
        query_dir : str,
        fingerprints_filepath : str,
        output_filepath : str,
        num_threads : int = 0,
    ):

    # load finger prints from disk
    with open(fingerprints_filepath, 'rb') as f:
        db_songs = pickle.load(f)
    
    query_files = sorted(glob.glob(os.path.join(query_dir, "*.wav")))

    print("Generating hashes from queries...")
    start = time.perf_counter()

    if num_threads > 0:
        with multiprocessing.Pool(processes=num_threads) as pool:
            query_songs_hashes = pool.map(fingerprint, query_files)
    else:
        query_songs_hashes = []
        for query_file in query_files:
            query_songs_hashes.append(fingerprint(query_file))
    stop = time.perf_counter()

    print(f"Fingerprinted {len(query_files)} queries in {stop-start:0.2f} s ({len(query_files)/(stop-start):0.1f} items/s)")

    query_matches = [] # we will store all matches here
    print("Finding matches...")
    start = time.perf_counter()
    for query_hashes in query_songs_hashes:
        query_matches.append(find_matches(query_hashes, db_songs))
    stop = time.perf_counter()

    print("-" * 64)
    metrics = compute_accuracy_metrics(query_files, query_matches, N=10)
    for key, val in metrics.items():
        print(f"{key} acc.  {val*100:0.2f}%")

    print()
    print(f"Analyized {len(query_files)} items in {stop-start:0.2f} s ({len(query_files)/(stop-start):0.1f} items/s)")
    print("-" * 64)

    # save the results to txt file
    with open(output_filepath, 'w') as f:
        for query_file, matches in zip(query_files, query_matches):

            # get the ground truth song id
            query_id = os.path.basename(query_file)
            f.write(f"{query_id}\t")
            song_id = query_id.split('-')[0]

            # sort the matches so that highest scoring are ranked first
            matches = [(db_song_id, score) for db_song_id, score in matches.items()]
            matches = sorted(matches, key=lambda a: a[1], reverse=True)

            for n in range(len(matches)): # write out the top three matches
                f.write(f"{matches[n][0]}.wav\t")
                if n > 2: break
            f.write("\n")

def baseline(
        database_dir : str,
        query_dir : str,
        num_threads : int = 0
    ):

    database_files = sorted(glob.glob(os.path.join(database_dir, "*.wav")))
    query_files = sorted(glob.glob(os.path.join(query_dir, "*.wav")))

    query_matches = []

    # create random queries 
    for query_file in query_files:
        rand_matches = np.random.permutation(database_files)
        rand_matches = [os.path.basename(f).replace('.wav','') for f in rand_matches]
        matches = {}
        for f in rand_matches:
            matches[f] = 0
        query_matches.append(matches)

    print("Baseline (random guessing)")
    print("-" * 64)
    metrics = compute_accuracy_metrics(query_files, query_matches, N=10)
    for key, val in metrics.items():
        print(f"{key} acc.  {val*100:0.2f}%")
    print("-" * 64)
    

if __name__ == '__main__':

    database_dir = 'data/database_recordings/'
    query_dir = 'data/query_recordings/'
    fingerprint_filepath = 'data/fingerprints.pkl'
    output_filepath = 'data/matches.txt'
    num_threads = 8 # adjust this based on CPU

    baseline(
        database_dir,
        query_dir
    )

    fingerprintBuilder(
                database_dir, 
                fingerprint_filepath,
                num_threads = num_threads
            )
            
    audioIdentification(
                query_dir, 
                fingerprint_filepath, 
                output_filepath,
                num_threads = num_threads
            )


