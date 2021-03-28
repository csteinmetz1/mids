import os
import glob
import time
import multiprocessing
import matplotlib.pyplot as plt

from mids.fingerprint import fingerprint, find_matches, compute_accuracy_metrics

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
    start = time.perf_counter()
    for database_file in database_files:
        params["audio_file"] = database_file
        db_songs_hashes.append(fingerprint(**params))
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

    start = time.perf_counter()
    for query_file in query_files:
        params["audio_file"] = query_file
        query_hashes = fingerprint(**params)
        query_matches.append(find_matches(query_hashes, db_songs))
    stop = time.perf_counter()
    query_time = stop-start
        
    metrics = compute_accuracy_metrics(query_files, query_matches)

    search_result = {
        "params" : params,
        "metrics" : metrics,
        "database_time" : database_time,
        "query_time" : query_time
    }

    return search_result

def plot_results(search_results, param_name="peak_distance", output_dir="data/search"):

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    fig, ax = plt.subplots(figsize=(8,6))

    acc = []
    param = []

    # extract the param and accuracy list
    for search_result in search_results:
        acc.append(search_result["metrics"]["Top-1"] * 100)
        param.append(search_result["params"][param_name])

    plt.scatter(param, acc)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel(param_name)
    plt.ylabel('Accuracy')
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, f'{param_name}.png'))
    plt.close('all')

if __name__ == '__main__':

    database_dir = 'data/database_recordings/'
    query_dir = 'data/query_recordings/'

    # minium peak distances for peak picking
    peak_distaces = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]

    starparams = zip([database_dir] * len(peak_distaces), 
                     [query_dir] * len(peak_distaces),
                     [{"peak_distance" : pd} for pd in peak_distaces])

    with multiprocessing.Pool(processes=12) as pool:
        search_results = pool.starmap(search, starparams)

    plot_results(search_results)

    