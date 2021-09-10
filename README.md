# Scripts
```build_video_index.py```
This module provides a cli for building and querying a faiss index of video clip embeddings.
The cli takes one of 2 possible commands, `build` or `query`.
For either command, the first parameter `-prefix` specifies the path to the faiss index file and the mapping file which  contains a dictionary of indices from the faiss file and `(filename, second_offset)`tuples.
For example, if the faiss files are located at data/indices/yt.faiss and data/indices/yt.mapping 
the command would be:
`python build_video_index.py -prefix data/indices/yt ...`

*see argparse for additional parameters*

``````