class ActiveLearners:
    RANDOM = "random"
    BALD = "bald"
    ALL = "all"


class Models:
    ERFNET = "erfnet"
    ERFNET_W_ALEATORIC = "erfnet_w_aleatoric"


class Losses:
    CROSS_ENTROPY = "xentropy"
    MSE = "mse"


IGNORE_INDEX = {"cityscapes": 19, "potsdam": 0, "flightmare": 9, "shapenet": None}


class Maps:
    # Federico original
    # MERGE = {
    #     0:0,  # ignore?
    #     1:1,
    #     2:2,
    #     3:3,
    #     4:1,
    #     5:0,
    #     6:0,
    #     7:0,
    #     8:0,
    #     9:0,
    #     10:0,
    #     11:0,
    #     12:12,
    #     13:13,
    #     14:13,
    #     15:12,
    #     16:16,
    #     17:16,
    #     18:1,
    #     255:0
    # }

    # CONTIGUOS = {
    #     0:0,
    #     1:1,
    #     2:2,
    #     3:3,
    #     12:4,
    #     13:5,
    #     16:6
    # }

    MERGE = {
        0: 0,  # ignore?
        1: 1,
        2: 2,
        3: 3,
        4: 1,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
        9: 0,
        10: 0,
        11: 0,
        12: 12,
        13: 2,
        14: 2,
        15: 12,
        16: 16,
        17: 16,
        18: 1,
        255: 0,
    }

    CONTIGUOS = {0: 0, 1: 1, 2: 2, 3: 3, 12: 4, 16: 5}
