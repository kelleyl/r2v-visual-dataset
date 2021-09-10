from build_video_index import main as main
import itertools
import json
from pyyoutube import Api
from typing import List
import os
import config
import random
import tqdm

recipes_file = "recipes.json"

api = Api(api_key=config.api_key)


def youtube_query_and_download():

    def get_video_id_list(query: str):
        r = api.search_by_keywords(
            q=query, search_type=["video"], count=10, limit=10, video_license="creativeCommon"
        )
        return [result.id.videoId for result in r.items]

    def download_videos_by_id(id_list: List[str], output_dir="data/youtube"):
        for video_id in id_list:
            os.system(
                f"youtube-dl -o {output_dir}/{video_id}.mp4 -f 135 http://youtube.com/watch?v={video_id}"
            )

    def load_recipe_titles(recipe_json_path: str):
        recipes = [json.loads(line) for line in open(recipe_json_path, "r")]
        return [recipe["name"] for recipe in recipes]

    current_recipe_id_list = [filename.strip(".mp4") for filename in os.listdir("data/youtube")]
    recipe_titles = load_recipe_titles(recipes_file)
    random.shuffle(recipe_titles) # todo: doing this right now to make sure we're not downloading the same queries every time, in the future we should have a list of queries we have run
    for title in tqdm.tqdm(recipe_titles):
        video_list = get_video_id_list(title + " recipe")
        print(f"downloading {len(set(video_list) - set(current_recipe_id_list))} new videos...")
        download_videos_by_id(video_list)

    video_id_list = set([video for query in recipe_titles for video in get_video_id_list(query)])
    download_videos_by_id(video_id_list)


def load_recipes_sentences_conllu(conllu_path: str) -> dict:
    with open(conllu_path, "r") as in_file:
        lines = in_file.readlines()
    lines = list(
        filter(lambda x: x.startswith("# sent_id") or x.startswith("# text"), lines)
    )
    id_text_dict = {}
    for i in range(0, len(lines), 2):
        sent_id = lines[i].split("=")[1].strip().replace("::", "_")
        sent_text = lines[i + 1].split("=")[1].strip()
        if "step" in sent_id:
            id_text_dict[sent_id] = sent_text
    return id_text_dict


def load_recipes_conllu(conllu_path: str) -> dict:
    id_query_dict = {}
    with open(conllu_path, "r") as in_file:
        lines = in_file.readlines()
    # split lines into recipes
    size = len(lines)
    idx_list = [idx for idx, val in enumerate(lines) if "newdoc" in val]
    res = [lines[i: j] for i, j in zip(idx_list, idx_list[1:] + ([size] if idx_list[-1] != size else []))]
    for recipe in res:
        reader = [line.split("\t") for line in recipe]
        event_indices = [idx for idx, val in enumerate(reader) if "B-EVENT" in val]
        text_between_events = [itertools.islice(reader, i, j) for i, j in zip(event_indices, event_indices[1:] + ([len(recipe)] if event_indices[-1] != len(recipe) else []))]
        for _id, tbe in enumerate(text_between_events):
            event_text = " ".join((conll_line[1] for conll_line in tbe if len(conll_line) > 2))
            event_id = f"{reader[0][0].split('=')[1].strip()}_{_id}"
            id_query_dict[event_id] = event_text
    return id_query_dict


def conllu_to_videos(conllu_path: str, video_output_dir: str) -> None:
    '''
    This function takes a path to a conllu recipe file and an output path to store the video results.
    '''
    id_text_dict = load_recipes_conllu(conllu_path)
    for k, v in id_text_dict.items():
        print(f"id:{k}, text:{v}")
        if not os.path.exists(f"data/{video_output_dir}"):
            os.mkdir(f"data/{video_output_dir}")
        main(
            [
                "-prefix",
                "data/indices/youtube_all", #todo un-hardcode this
                "query",
                v,
                "-output",
                f"data/{video_output_dir}/{k}",
            ]
        )


# def videos_to_keyframes(video_dir:str, keyframe_outout_dir:str) -> None:
#     """This function takes a directory containing directories of videos and makes a new directory in which the original
#     folder structure is maintained, but each video is converted to a directory containing 3 individual frames and the
#     result of stitching the 3 frames together"""
#     if not os.path.exists(keyframe_outout_dir):
#         os.mkdir("keyframe_output_dir")
#     for r, d, f in os.walk(video_dir):
#         for filename in f:
#             if filename.endswith('.mp4') or filename.endswith('.mkv') or filename.endswith('.webm'):
#                 youtube_id = os.path.splitext(filename)[0]
#                 if not os.path.exists(os.path.join(keyframe_outout_dir, youtube_id)):
#                     os.mkdir(os.path.join(keyframe_outout_dir, youtube_id))


if __name__ == "__main__":
    youtube_query_and_download()
