import warnings
from math import ceil
from pathlib import Path

import cv2
import numpy as np
from einops import repeat
from tqdm import tqdm

from .util import apply_filters
from .util import discretize
from .util import filter_frames_index_function
from .util import generate_frames


def add_black_and_grey_screen(
    writer: cv2.VideoWriter,
    xsize: int,
    ysize: int,
    fps: int,
):
    # add 2s of black screen
    black_frame = np.full(
        shape=(ysize, xsize, 3),
        fill_value=0,
        dtype=np.uint8,
    )
    for _ in tqdm(range(2 * fps), desc="black screen"):
        writer.write(black_frame)
    # add 2s of grey screen
    grey_frame = np.full(
        shape=(ysize, xsize, 3),
        fill_value=255 / 2,
        dtype=np.uint8,
    )
    for _ in tqdm(range(2 * fps), desc="grey screen"):
        writer.write(grey_frame)


def zebra_noise(
    output_file: Path,
    xsize: int,
    ysize: int,
    tdur: int,
    levels: int = 10,
    xyscale: float = 0.2,
    tscale: int = 50,
    fps: int = 30,
    xscale: float = 1.0,
    yscale: float = 1.0,
    seed: int = 0,
    filters: list = [("comb", 0.08)],
):
    """Generate a .mp4 of zebra noise.

    This method is a simplified interface for the PerlinStimulus class, designed to only generate zebra noise
    as defined in the paper.

    Parameters
    ----------
    output_file : string
        Filename to save the generated .mp4 file
    xsize, ysize : int
        The x and y dimensions of the output video. (Sometimes these will be rounded up to multiples of 16.)
    tdur : float
        The duration of the video in seconds
    levels : int
        The number of octaves to use when approximating the 1/f spectrum. The default of 10 should be more than enough.
    xyscale : float from (0,1)
        The spatial scale of the Perlin noise. Values near 0 will make the video smoother and near 1 choppier.
    tscale : int
        A scaling factor to set the speed of the video
    xscale, yscale : float
        Resize the x and y dimensions of the output
    fps : int
        Frames per second
    seed : int
        Random seed

    Returns
    -------
    None, but saves the video file to the desired filename
    """
    tsize = int(tdur * fps)
    tscale = tscale * (fps / 30)
    textra = (tscale - (tsize % tscale)) % tscale
    if textra > 0:
        warnings.warn(
            f"Adding {textra} extra timepoints to make tscale a multiple of tdur"
        )
    tsize += round(textra) if (textra % 1 < 1e-5) else ceil(textra)
    get_index = filter_frames_index_function(filters, tsize)

    writer = cv2.VideoWriter(
        filename=str(output_file),
        fourcc=cv2.VideoWriter_fourcc(*"MJPG"),
        fps=fps,
        frameSize=(xsize, ysize),
    )

    add_black_and_grey_screen(writer=writer, fps=fps, xsize=xsize, ysize=ysize)

    for _i in tqdm(range(0, tsize), desc="Zebra noise"):
        i = get_index(_i)
        frame = generate_frames(
            xsize,
            ysize,
            tsize,
            [i],
            levels=levels,
            xyscale=xyscale,
            tscale=tscale,
            xscale=xscale,
            yscale=yscale,
            seed=seed,
        )
        # TODO I don't think this will work with the photodiode filter
        filtered = apply_filters(frame[None], filters)[0]
        disc = discretize(filtered[:, :, 0])
        writer.write(repeat(disc, "h w -> h w c", c=3))
    writer.release()
