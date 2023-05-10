import pandas as pd
import numpy as np

def confusion_matrix(actual, pred, n_classes) -> pd.DataFrame:

   
    mat = pd.crosstab(actual, pred, normalize = False)
    
    # sometimes a column might be missing
    expected_index = [i for i in range(n_classes)]
    mat = mat.reindex(index = expected_index,
                      columns = expected_index,
                      fill_value=0) 
    return mat

def apply_names(conf_matrix: pd.DataFrame) -> None:
    n_classes = len(conf_matrix)
    if n_classes == 2:
        names = ["Defect", "No Defect"]
    else:
        names = [f"Defect {i}" for i in range(1, n_classes)] + ["No Defect"]
    conf_matrix.columns = names
    conf_matrix.index = names
    return conf_matrix

def normalize_confusion(conf_matrix: pd.DataFrame):
    conf_matrix = conf_matrix.copy()
    for i in range(len(conf_matrix)):
        s = sum(conf_matrix[i])
        if s > 0:
            conf_matrix[i] = conf_matrix[i] / s
    
    return conf_matrix

def metric_iou(conf_matrix: pd.DataFrame) -> pd.DataFrame:
    """
        conf_matrix must be NON NORMALIZED.

        Returns Intersection over Union score for each class, which is 
        calculated as 

            (true positives) / (true positives + false positives + false negatives)
    """

    n_classes = len(conf_matrix)
    ious = np.array([])

    for i in range(n_classes):
        intersection = conf_matrix[i][i]
        union = np.sum(np.array(conf_matrix)[:,i]) + np.sum(conf_matrix[i]) - conf_matrix[i][i]

        ious = np.append(ious, intersection / union)

    return ious