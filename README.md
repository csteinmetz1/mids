# mids: Music IDentification System
Implementation of content-based audio search algorithm.

# Usage
Create a virtual environment and install requirements.
```
python3 -m venv env/
source env/bin/activate
pip install -r requirements.txt
```

Run the code which will create the database and check all queries.
```
python fingerprint.py
```

# Ideas

## Improving accuracy
- Revisit how the spectrograms are getting computed. 
- Use multiple STFTs at different frame sizes.
- Build a better peak picking algorithm.
- Run a hyperparameter optimization grid (or random) search across parameter space.

## Evaluation
- Create synthetic data by applying some augmentations on database recordings.
- Figure out which songs are being missed by the system, and inspect peaks.
- Print some statistics for how many peaks are found in each, and how many peaks per target zone. 
- Create a pareto curve that shows the accuracy and compute time for different methods.