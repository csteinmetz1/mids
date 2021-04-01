# mids: Music IDentification System
Implementation of content-based audio search algorithm.

# Setup & Usage
Create a virtual environment and install requirements.
```
python3 -m venv env/
source env/bin/activate
pip install -r requirements.txt
```

Ensure that you have downloaded both the database recordings and query recordings. 
These can be downloaded as follows, and then placed in the `data/` directory. 
```
wget -O database.zip https://collect.qmul.ac.uk/down?t=R8SDLMOKUOSCD2VB/6P63FFT4AN0581R7V49FJKO 
wget -O query.zip https://collect.qmul.ac.uk/down?t=450TPH3RDUJNA920/6P4TNTJT7GSTR7NUC226IJ8
unzip database.zip 
unzip query.zip
mkdir data
mv database_recordings/ data/
mv query_recordings/ data/
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