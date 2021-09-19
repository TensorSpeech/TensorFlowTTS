import os
import shutil
from tqdm import tqdm
import argparse

from scipy.ndimage import zoom
from skimage.data import camera
import numpy as np
from scipy.spatial.distance import cdist


def safemkdir(dirn):
    if not os.path.isdir(dirn):
        os.mkdir(dirn)


from pathlib import Path


def duration_to_alignment(in_duration):
    total_len = np.sum(in_duration)
    num_chars = len(in_duration)

    attention = np.zeros(shape=(num_chars, total_len), dtype=np.float32)
    y_offset = 0

    for duration_idx, duration_val in enumerate(in_duration):
        for y_val in range(0, duration_val):
            attention[duration_idx][y_offset + y_val] = 1.0

        y_offset += duration_val

    return attention


def rescale_alignment(in_alignment, in_targcharlen):
    current_x = in_alignment.shape[0]
    x_ratio = in_targcharlen / current_x
    pivot_points = []

    zoomed = zoom(in_alignment, (x_ratio, 1.0), mode="nearest")

    for x_v in range(0, zoomed.shape[0]):
        for y_v in range(0, zoomed.shape[1]):
            val = zoomed[x_v][y_v]
            if val < 0.5:
                val = 0.0
            else:
                val = 1.0
                pivot_points.append((x_v, y_v))

            zoomed[x_v][y_v] = val

    if zoomed.shape[0] != in_targcharlen:
        print("Zooming didn't rshape well, explicitly reshaping")
        zoomed.resize((in_targcharlen, in_alignment.shape[1]))

    return zoomed, pivot_points


def gather_dist(in_mtr, in_points):
    # initialize with known size for fast
    full_coords = [(0, 0) for x in range(in_mtr.shape[0] * in_mtr.shape[1])]
    i = 0
    for x in range(0, in_mtr.shape[0]):
        for y in range(0, in_mtr.shape[1]):
            full_coords[i] = (x, y)
            i += 1

    return cdist(full_coords, in_points, "euclidean")


def create_guided(in_align, in_pvt, looseness):
    new_att = np.ones(in_align.shape, dtype=np.float32)
    # It is dramatically faster that we first gather all the points and calculate than do it manually
    # for each point in for loop
    dist_arr = gather_dist(in_align, in_pvt)
    # Scale looseness based on attention size. (addition works better than mul). Also divide by 100
    # because having user input 3.35 is nicer
    real_loose = (looseness / 100) * (new_att.shape[0] + new_att.shape[1])
    g_idx = 0
    for x in range(0, new_att.shape[0]):
        for y in range(0, new_att.shape[1]):
            min_point_idx = dist_arr[g_idx].argmin()

            closest_pvt = in_pvt[min_point_idx]
            distance = dist_arr[g_idx][min_point_idx] / real_loose
            distance = np.power(distance, 2)

            g_idx += 1

            new_att[x, y] = distance

    return np.clip(new_att, 0.0, 1.0)


def get_pivot_points(in_att):
    ret_points = []
    for x in range(0, in_att.shape[0]):
        for y in range(0, in_att.shape[1]):
            if in_att[x, y] > 0.8:
                ret_points.append((x, y))
    return ret_points


def main():
    parser = argparse.ArgumentParser(
        description="Postprocess durations to become alignments"
    )
    parser.add_argument(
        "--dump-dir",
        default="dump",
        type=str,
        help="Path of dump directory",
    )
    parser.add_argument(
        "--looseness",
        default=3.5,
        type=float,
        help="Looseness of the generated guided attention map. Lower values = tighter",
    )
    args = parser.parse_args()
    dump_dir = args.dump_dir
    dump_sets = ["train", "valid"]

    for d_set in dump_sets:
        full_fol = os.path.join(dump_dir, d_set)
        align_path = os.path.join(full_fol, "alignments")

        ids_path = os.path.join(full_fol, "ids")
        durations_path = os.path.join(full_fol, "durations")

        safemkdir(align_path)

        for duration_fn in tqdm(os.listdir(durations_path)):
            if not ".npy" in duration_fn:
                continue

            id_fn = duration_fn.replace("-durations", "-ids")

            id_path = os.path.join(ids_path, id_fn)
            duration_path = os.path.join(durations_path, duration_fn)

            duration_arr = np.load(duration_path)
            id_arr = np.load(id_path)

            id_true_size = len(id_arr)

            align = duration_to_alignment(duration_arr)

            if align.shape[0] != id_true_size:
                align, points = rescale_alignment(align, id_true_size)
            else:
                points = get_pivot_points(align)

            if len(points) == 0:
                print("WARNING points are empty for", id_fn)

            align = create_guided(align, points, args.looseness)

            align_fn = id_fn.replace("-ids", "-alignment")
            align_full_fn = os.path.join(align_path, align_fn)

            np.save(align_full_fn, align.astype("float32"))


if __name__ == "__main__":
    main()
