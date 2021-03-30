import os
import numpy as np

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