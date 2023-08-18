import os
import numpy as np
import glob
import imageio

LABELS = {
    "shapenet": {
        "background": {"color": (255, 255, 255), "id": 0},
        "car": {"color": (102, 102, 102), "id": 1},
        "chair": {"color": (0, 0, 255), "id": 2},
        "table": {"color": (0, 255, 255), "id": 3},
        "sofa": {"color": (255, 0, 0), "id": 4},
        "airplane": {"color": (102, 0, 204), "id": 5},
        "camera": {"color": (0, 102, 0), "id": 6},
        "birdhouse": {"color": (255, 153, 204), "id": 7},
    }
}


def label2color(label_map, theme="shapenet"):
    assert theme in LABELS.keys()
    rgb = np.zeros((label_map.shape[0], label_map.shape[1], 3), np.int8)
    for _, cl in LABELS[theme].items():  # loop each class label
        if cl["color"] == (0, 0, 0):
            continue  # skip assignment of only zeros
        mask = np.where(label_map == cl["id"], 1, 0)
        rgb[:, :, 0] += mask * cl["color"][0]
        rgb[:, :, 1] += mask * cl["color"][1]
        rgb[:, :, 2] += mask * cl["color"][2]
    return rgb


dataset_path = "../dataset/shapenet/"
scene_list = [
    name for name in os.listdir(dataset_path) if os.path.isdir(f"{dataset_path}/{name}")
]
scene_path = [os.path.join(dataset_path, scene) for scene in scene_list]
for scene in scene_path:
    print(scene)
    semantic_path = os.path.join(scene, "semantics")
    npy_paths = [
        x for x in glob.glob(os.path.join(semantic_path, "*")) if (x.endswith(".npy"))
    ]
    npy_paths = sorted(npy_paths)

    for i, inst in enumerate(npy_paths):
        label = np.load(inst)
        # for visualizing semantic label
        semantic_map = label2color(label).astype("uint8")
        # semantic_map = np.round(semantic_map).astype("uint8")
        imageio.imwrite(f"{semantic_path}/{i+1:04d}.png", semantic_map)
