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
