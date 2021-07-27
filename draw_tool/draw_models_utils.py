import torch
num_classes = 9

if num_classes == 9:
    lc_pixels = {
    0: str([255, 255, 255, 255]),
    1: str([210, 0, 0, 255]),
    2: str([253, 211, 39, 255]),
    3: str([35, 152, 0, 255]),
    4: str([8, 98, 0, 255]),
    5: str([249, 150, 39, 255]),
    6: str([141, 139, 0, 255]),
    7: str([149, 107, 196, 255]),
    8: str([77, 37, 106, 255]),
}

    lc_labels_classes = {
        0: 'Other',
        1: 'Artificial surfaces and constructions',
        2: 'Cultivated areas',
        3: 'Broadleaf tree cover',
        4: 'Coniferous tree cover',
        5: 'Herbaceous vegetation',
        6: 'Moors and Heathland',
        7: 'Marshes',
        8: 'Peatbogs',
    }
    possible_lc_classes = {
        str([255, 255, 255]): 0,
        str([210, 0, 0]): 1,
        str([253, 211, 39]): 2,
        str([35, 152, 0, ]): 3,
        str([8, 98, 0, ]): 4,
        str([249, 150, 39, ]): 5,
        str([141, 139, 0, ]): 6,
        str([149, 107, 196, ]): 7,
        str([77, 37, 106, ]): 8,
        str([20, 69, 249,]): 0,
        str([106, 255, 255,]): 0,
        str([154, 154, 154,]): 0,
        str([95, 53, 6]): 0,
        str([176, 91, 16]): 0,
    }
else:
    possible_lc_classes = {
        str([255, 255, 255]): 0,
        str([210, 0, 0]): 1,
        str([253, 211, 39]): 2,
        str([176, 91, 16]): 3,
        str([35, 152, 0]): 4,
        str([8, 98, 0]): 5,
        str([249, 150, 39]): 6,
        str([141, 139, 0]): 7,
        str([95, 53, 6]): 8,
        str([149, 107, 196,]): 9,
        str([77, 37, 106,]): 10,
        str([154, 154, 154,]): 11,
        str([106, 255, 255,]): 12,
        str([20, 69, 249,]): 13,
    }

tensor_type = torch.FloatTensor