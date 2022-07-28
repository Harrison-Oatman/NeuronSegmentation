import json
from ..const import *
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes
from scipy.ndimage import distance_transform_edt
import skan
from collections import defaultdict
from tqdm import tqdm


def get_center_val(im: np.ndarray):
    centers = np.array(im.shape) // 2
    return im[centers[0], centers[1], :]


def get_connected_components(skel: skan.csr.Skeleton, branch):
    paths = skel.paths_list()
    b_to_e = [[path[0], path[-1]] for path in paths]
    e_to_b = defaultdict(list)
    for i, path in enumerate(paths):
        e_to_b[path[0]].append(i)
        e_to_b[path[-1]].append(i)

    stack = list(b_to_e[int(branch)])
    visited = set(stack)
    pairs = set()
    in_branches = set(branch)

    while len(stack) > 0:
        pt = stack.pop()
        contained_branches = e_to_b[pt]
        if len(contained_branches) < 2:
            continue

        for i, a in enumerate(contained_branches):
            for b in contained_branches[i + 1:]:
                pairs.add((pt, tuple(sorted([a, b]))))

            for new_pt in b_to_e[a]:
                if new_pt not in visited:
                    stack.append(new_pt)
                    visited.add(new_pt)

            in_branches.add(a)

    return pairs, in_branches


def segment_image(pom, i=0):
    thresholded = pom > 0
    thresholded = remove_small_objects(thresholded, min_size=100)
    thresholded = remove_small_holes(thresholded, area_threshold=25)
    skeletonized = skeletonize(thresholded)

    if np.sum(skeletonized) < 5*255:
        return [], [], None, None

    skel = skan.csr.Skeleton(skeletonized, source_image=pom)

    branch_labeled = np.array(skel.path_label_image())
    nearest_branch = distance_transform_edt(branch_labeled == 0,
                                            return_distances=False, return_indices=True)
    nearest_val = branch_labeled[nearest_branch[0], nearest_branch[1], nearest_branch[2]]

    root_branch = get_center_val(nearest_val)

    pairs, branches = get_connected_components(skel, root_branch - 1)
    pairs = list((skel.coordinates[p[0]], (p[1][0] + 1, p[1][1] + 1)) for p in pairs)
    branches = list(branch + 1 for branch in branches)
    return pairs, branches, branch_labeled, nearest_val


def load_branches(json_path, out_path):
    with open(json_path) as f:
        datasets = dict(json.loads(f.read()))

    sources = [key for key in datasets.keys()]

    for source in sources:
        examples = datasets[source]
        for i, example in tqdm(enumerate(examples)):
            pom = np.load(example['inference'])
            pairs, branches, branch_labeled, branch_seg = segment_image(pom, i)

            pairs = [([int(a) for a in p[0]], [int(a) for a in p[1]]) for p in pairs]
            branches = [int(b) for b in branches]

            branch_im_path = INFERENCE_OUTPUT_PATH + "examples\\" + f"branch_im_{source}_{i}.npy"
            branch_seg_path = INFERENCE_OUTPUT_PATH + "examples\\" + f"branch_seg_{source}_{i}.npy"

            example['branch_im'] = branch_im_path
            example['branch_seg'] = branch_seg_path
            example['pairs'] = pairs
            example['branches'] = branches

            np.save(branch_im_path, branch_labeled)
            np.save(branch_seg_path, branch_seg)

            examples[i] = example
        datasets[source] = examples

    jsons = json.dumps(datasets)
    with open(out_path, "w") as f:
        f.write(jsons)


def main():
    json_path = INFERENCE_OUTPUT_PATH + "data.json"
    processed_path = INFERENCE_OUTPUT_PATH + "processed.json"
    load_branches(json_path, processed_path)


if __name__ == '__main__':
    main()
