import ffmpeg
import os
import numpy as np
import torch as th


def get_video_start(video_path, start, output_directory=None):
    """This function was adapted from  https://github.com/antoine77340/MIL-NCE_HowTo100M
    """
    num_frames = 32
    fps = 8
    num_sec = num_frames / float(fps)
    size = 224
    cmd = (
        ffmpeg.input(video_path, ss=start, t=num_sec + 0.1).filter('fps', fps=fps)
    )
    aw, ah = 0.5, 0.5
    cmd = (
        cmd.crop('(iw - min(iw,ih))*{}'.format(aw),
                 '(ih - min(iw,ih))*{}'.format(ah),
                 'min(iw,ih)',
                 'min(iw,ih)').filter('scale', size, size)
    )
    try:
        if output_directory:
            base_name = os.path.basename(video_path)[:15]
            if not os.path.exists(f'{output_directory}'):
                os.mkdir(f'{output_directory}')
            output_filename = f'{output_directory}/{base_name}_{start}.mp4'
            if os.path.exists(output_filename):
                os.remove(output_filename)
            if not os.path.exists(output_filename):
                out2, _ = (
                    cmd.output(output_filename).run(capture_stdout=True, quiet=True)
                )
        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24').run(capture_stdout=True, quiet=True)
        )
    except Exception as e:
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))
        raise e
    video = np.frombuffer(out, np.uint8).reshape([-1, size, size, 3])
    video = th.from_numpy(video)
    video = video.permute(3, 0, 1, 2)
    if video.shape[1] < num_frames:
        print("padding with zeros")
        zeros = th.zeros((3, num_frames - video.shape[1], size, size), dtype=th.uint8)
        video = th.cat((video, zeros), axis=1)
    return video[:, :num_frames]


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
    res = [
        lines[i:j]
        for i, j in zip(
            idx_list, idx_list[1:] + ([size] if idx_list[-1] != size else [])
        )
    ]
    for recipe in res:
        reader = [line.split("\t") for line in recipe]
        event_indices = [idx for idx, val in enumerate(reader) if "B-EVENT" in val]
        text_between_events = [
            itertools.islice(reader, i, j)
            for i, j in zip(
                event_indices,
                event_indices[1:]
                + ([len(recipe)] if event_indices[-1] != len(recipe) else []),
                )
        ]
        for _id, tbe in enumerate(text_between_events):
            event_text = " ".join(
                (conll_line[1] for conll_line in tbe if len(conll_line) > 2)
            )
            event_id = f"{reader[0][0].split('=')[1].strip()}_{_id}"
            id_query_dict[event_id] = event_text
    return id_query_dict


def conllu_to_videos(conllu_path: str, video_output_dir: str) -> None:
    """
    This function takes a path to a conllu recipe file and an output path to store the video results.
    """
    from video_index import main as main

    id_text_dict = load_recipes_conllu(conllu_path)
    for k, v in id_text_dict.items():
        print(f"id:{k}, text:{v}")
        if not os.path.exists(f"data/{video_output_dir}"):
            os.mkdir(f"data/{video_output_dir}")
        main(
            [
                "-prefix",
                "data/indices/youtube_all",  # todo un-hardcode this
                "query",
                v,
                "-output",
                f"data/{video_output_dir}/{k}",
            ]
        )
