import json
from pyyoutube import Api
from typing import List, Tuple
import os
import config
import tqdm

LOG_FILE = "queries.log"
RECIPES_FILE = "recipes.json"
DOWNLOAD_DIRECTORY = os.path.join("data", "youtube")
# DOWNLOAD_DIRECTORY = config.download_directory
api = Api(api_key=config.api_key)


def log_query(query_json: dict, log_file: str = LOG_FILE) -> None:
    with open(log_file, "a") as log:
        json.dump(query_json, log)
        log.write("\n")


def load_query_log(log_file: str = LOG_FILE) -> Tuple[List[str], List[str], List[str]]:
    queries = [json.loads(line) for line in open(log_file, "r")]
    video_history = [log_line["video_id"] for log_line in queries]
    recipe_history = [log_line["recipe_id"] for log_line in queries]
    query_history = [log_line["query_text"] for log_line in queries]
    return video_history, recipe_history, query_history


def load_video_id_list(directory: str = DOWNLOAD_DIRECTORY) -> List[str]:
    return [f.strip(".mp4") for f in os.listdir(directory)]


def get_video_id_list(query: str) -> List[str]:
    r = api.search_by_keywords(
        q=query,
        search_type=["video"],
        count=10,
        limit=10,
        video_license="creativeCommon",
    )
    return [result.id.videoId for result in r.items]


def download_videos_by_id(id_list: List[str], output_dir=DOWNLOAD_DIRECTORY) -> None:
    for video_id in id_list:
        if video_id not in load_video_id_list():
            os.system(
                f"youtube-dl -o {os.path.join(output_dir,video_id)}.mp4 -f 135 http://youtube.com/watch?v={video_id}"
            )


def load_recipes(recipe_json_path: str = RECIPES_FILE):
    return [json.loads(line) for line in open(recipe_json_path, "r")]


def youtube_query_and_download():
    current_recipe_id_list = load_video_id_list()
    recipes = load_recipes()
    for recipe in tqdm.tqdm(recipes):
        video_query = f"{recipe['name']} recipe"
        _, _, query_history = load_query_log()
        if video_query in query_history:
            print(f"skipping {video_query}")
        else:
            video_list = get_video_id_list(video_query)
            print(
                f"downloading {len(set(video_list) - set(current_recipe_id_list))} new videos..."
            )
            for video in video_list:
                log_json = {
                    "video_id": video,
                    "recipe_id": recipe["id"],
                    "query_text": video_query,
                }
                log_query(log_json)
            if len(video_list) == 0:   # placeholder
                log_json = {"video_id":  None,
                            "recipe_id": recipe["id"],
                            "query_text": video_query
                            }
                log_query(log_json)

            download_videos_by_id(video_list)


if __name__ == "__main__":
    youtube_query_and_download()
