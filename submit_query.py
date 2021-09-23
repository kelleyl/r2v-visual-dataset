"""
Submits query and saves video frames to "outputs/" directory if SAVE_FRAMES=True.
This should be run on tarski.
"""
import os

if __name__ == "__main__":
    QUERY="Add the egg and mix well."
    PREFIX="youtube_all"
    SAVE_FRAMES=True
    os.system(f'python build_video_index.py -prefix /home/kmlynch/projects/r2v-visual-data-tools/data/indices/{PREFIX} query "{QUERY}" -save_frames {SAVE_FRAMES}')
