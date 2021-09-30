# Scripts
```yt_download.py```
This module contains the functions to load recipes from json and query youtube for recipes with
the given recipe title. The downloader requires that an api key be included in a `config.py` file.
The `config.py` file should contain a line like the following `api_key="XXXXXX"` (with the user's api key inserted). The
api key can be generated from the google developer console.

```video_index.py```
This module provides a cli for building and querying a faiss index of video clip embeddings.
The cli takes one of 2 possible commands, `build` or `query`.
For either command, the first parameter `-prefix` specifies the path to the faiss index file and the mapping file which  contains a dictionary of indices from the faiss file and `(filename, second_offset)`tuples.
For example, if the faiss files are located at data/indices/yt.faiss and data/indices/yt.mapping 
the command would be:
`python build_video_index.py -prefix data/indices/yt ...`

*see argparse for additional parameters*

## Getting Image Frames from a Query 
The scripts ```submit_query.py``` and ```retrieve_output.sh``` can be run in tandem on Tarski and locally in order to quickly test the retrieval results on a specific data source. 

In ```submit_query```, an example query is formatted using the set variables. Run this on Tarski, and an ```outputs/``` directory will be created containing the frame results. 

Then, modify the variables in ```retrieve_output.sh``` with your Tarski username and the directory where ```submit_query.py``` was run on Tarski. Running ```./retrieve_output.sh``` locally will then call scp to download the contents of the Tarski ```outputs/``` directory to a local location.
``````