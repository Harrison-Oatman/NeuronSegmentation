import re
import numpy as np
import matplotlib.pyplot as plt


def branch_labels(process_names=None, process_image=None, **kwargs):
    branch_index = 1
    branches = {}
    process_to_branch = {0: 0}

    for i, name in enumerate(process_names[1:]):
        i += 1

        # determine branch identifier by reading up to . and the following digit
        branch_name = re.match(r'\d*\.\d', name)[0]

        branch_num = branches.get(branch_name)

        if branch_num is None:
            branch_num = branch_index
            branches[branch_name] = branch_num
            branch_index += 1

        process_to_branch[i] = branch_num

    def process_map(i: int) -> int:
        return process_to_branch[i]

    branch_image = np.vectorize(process_map)(process_image)

    print(branch_image.dtype)

    # plt.imshow(branch_image)
    # plt.show()

    return branch_image
