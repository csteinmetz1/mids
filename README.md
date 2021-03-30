# mids: Music IDentification System
Implementation of content-based audio search algorithm.

# Setup & Usage
Create a virtual environment and install requirements.
```
python3 -m venv env/
source env/bin/activate
pip install -r requirements.txt
```

The `run.py` script will generate fingerprints for the database, 
and then generate fingerprints for all queries, saving top 3 matches to a `.txt` file.
Ensure that the paths are correct at the bottom the script, 
and point to the correct directories for the database and query `.wav` files.
```
python run.py
```

The `search.py` script was used to generate plots over various 
configurations of the fingerprinting parameters. 
```
python search.py
```