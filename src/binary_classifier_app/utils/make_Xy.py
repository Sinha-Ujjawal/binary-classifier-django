from typing import List, Tuple


def make_Xy(
    *, type_0: List[List[int]], type_1: List[List[int]]
) -> Tuple[List[List[int]], List[int]]:
    X = []
    y = []
    for label, coords in enumerate([type_0, type_1]):
        for coord in coords:
            X.append(coord)
            y.append(label)
    return X, y
