import tqdm as tqdm
import faiss
import pickle
import traceback
import tensorflow as tf
import tensorflow_hub as hub
from utils import *

import os ##todo 6/29/21 kelleylynch this fix shouldnt be necessary
os.environ['KMP_DUPLICATE_LIB_OK']='True'


hub_handle = 'https://tfhub.dev/deepmind/mil-nce/s3d/1'
model = hub.load(hub_handle)


def get_video_embedding(video_frames):
    video_frames = video_frames.numpy().reshape((1,32,224,224,3))/255.0 # torch to tensorflow
    vision_output = model.signatures['video'](tf.constant(tf.cast(video_frames, dtype=tf.float32)))
    return vision_output["video_embedding"]


def generate_video_embeddings(video_path_list, start_seconds=0):
    video_embedding_list = []
    video_id_dict = {}
    for video in video_path_list:
        if os.path.isfile(video):
            video_abs_path = os.path.abspath(video)
            try:
                if video_abs_path.endswith(".mkv"):
                    seconds_duration_string = ffmpeg.probe(video_abs_path)["streams"][0]["metadata"]["duration"]
                else:
                    seconds_duration_string = ffmpeg.probe(video_abs_path)["streams"][0]["duration"] ##todo 6/28/21 kelleylynch  in order to support mkv format, need to get duration from within metadata
                seconds_duration = int(seconds_duration_string[:seconds_duration_string.index(".")])
                for start in range(start_seconds, seconds_duration, 4):
                    video_frames = get_video_start(video_abs_path, start=start)
                    video_output = get_video_embedding(video_frames)
                    video_embedding_list.append(video_output.numpy())
                    video_id_dict[len(video_embedding_list)-1] = (video, start)
            except Exception as e:
                print(video_abs_path)
                traceback.print_exc()
    video_embeddings = np.concatenate(video_embedding_list, axis=0)
    return video_embeddings, video_id_dict


def save_to_faiss_index(new_embeddings, faiss_path, new_id_dict, id_dict_path):
    if os.path.exists(faiss_path):  # add to existing index
        index = faiss.read_index(faiss_path)
        with open(id_dict_path, 'rb') as pkl:
            mapping_dict = pickle.load(pkl)
            start_id = len(mapping_dict)
            for key, value in new_id_dict.items():
                mapping_dict[start_id+key] = value
    else:  # create new index
        index = faiss.IndexFlatIP(new_embeddings.shape[1])
        mapping_dict = new_id_dict
    index.add(new_embeddings)
    faiss.write_index(index, faiss_path)
    with open(id_dict_path, 'wb') as out:
        pickle.dump(mapping_dict, out)


class IndexSearch:
    def __init__(self, index, mapping_dict, model=model):
        self.index = index
        self.mapping_dict = mapping_dict
        self.model = model

    def video_id_list(self):
        """returns a list of youtube ids that have been indexed"""
        return set([os.path.splitext(os.path.basename(x[0]))[0] for x in self.mapping_dict.values()])

    def query_index(self, query_string, k=5, save_to_directory=None):
        text_embedding = self.model.signatures['text'](tf.constant(np.array([query_string])))["text_embedding"]
        d, i = self.index.search(np.array(text_embedding).astype(np.float32), k=5) ##todo 9/1/21 kelleylynch why cant k be passed here as a variable
        result_list = []
        for result in i[0]:
            filename, seconds = self.mapping_dict[result]
            result_list.append(self.mapping_dict[result])
            if save_to_directory:
                get_video_start(filename, seconds, save_to_directory)
        return result_list


def load_index_files(prefix):
    index = faiss.read_index(f"{prefix}.faiss")
    with open(f"{prefix}.mapping", 'rb') as mapping_file:
        id_dict = pickle.load(mapping_file)
    return index, id_dict


def main(args=None):
    def build(args):
        previously_indexed_file_list = []
        if os.path.exists(f"{args.prefix}.mapping"):
            index, id_dict = load_index_files(args.prefix)
            search_obj = IndexSearch(index, id_dict)
            previously_indexed_file_list = search_obj.video_id_list()
            print(f"adding new embeddings to existing index: {args.prefix}")

        video_list = []
        for r, d, f in os.walk(args.input):
            for filename in f:
                if filename.endswith('.mp4') or filename.endswith('.mkv') or filename.endswith('.webm'):
                    if os.path.splitext(filename)[0] not in previously_indexed_file_list:
                        video_list.append(os.path.join(r, filename))

        print(f"generating embeddings for {len(video_list)} files")
        for batch in tqdm.tqdm(range(0, len(video_list), 10)):
            video_batch = video_list[batch:batch+10]
            if not args.dryrun:
                embeds, id_dict = generate_video_embeddings(video_batch)
                save_to_faiss_index(embeds, f"{args.prefix}.faiss", id_dict, f"{args.prefix}.mapping")

    def query(args):
        index, id_dict = load_index_files(args.prefix)
        search_obj = IndexSearch(index, id_dict)
        result_list = search_obj.query_index(args.query_string, args.output)
        print(result_list)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-prefix", required=True, help="filename prefix for faiss index and mapping dict")
    subparsers = parser.add_subparsers()

    build_parser = subparsers.add_parser("build", help="build faiss index and mapping dictionary from video directory")
    build_parser.add_argument("-input", required=True, help="root directory of video input")
    build_parser.add_argument("-dryrun", action="store_true")
    build_parser.set_defaults(func=build)

    query_parser = subparsers.add_parser("query", help="query the index for a given string")
    query_parser.add_argument("query_string", type=str)
    query_parser.add_argument("-output", default=None, help="directory to store the output, if not supplied, video paths will be printed to stdout")
    query_parser.add_argument("-count", help="number of videos to return", type=int, choices=range(1, 5))
    query_parser.set_defaults(func=query)

    args = parser.parse_args(args)
    args.func(args)


if __name__ == '__main__':
    main()


