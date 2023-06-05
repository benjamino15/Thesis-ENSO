import numpy as np

def comparison_metrics(reference_val_matrix, reference_p_matrix, val_matrix, p_matrix, threshold):
    mask_array = np.where(reference_p_matrix < threshold, 1, 0)
    mask_array = mask_array.flatten().tolist()
    reference_val_matrix = reference_val_matrix.flatten().tolist()
    val_matrix = val_matrix.flatten().tolist()
    abs_difference = []
    squared_difference = []
    for i in range(len(mask_array)):
        if mask_array[i] == 1:
            abs_difference.append(abs(reference_val_matrix[i] - val_matrix[i]))
            squared_difference.append((reference_val_matrix[i] - val_matrix[i])**2)

    mask_array2 = np.where(p_matrix < threshold, 1, 0)
    mask_array2 = mask_array2.flatten().tolist()
    true_positives = []
    false_positives = []
    false_negatives = []
    for i in range(len(mask_array)):
        if mask_array[i] == 1 and mask_array2[i] == 1:
            true_positives.append(1)
        elif mask_array[i] == 1 and mask_array2[i] == 0:
            false_negatives.append(1)
        elif mask_array[i] == 0 and mask_array2[i] == 1:
            false_positives.append(1)

    precision = np.sum(true_positives)/(np.sum(true_positives) + np.sum(false_positives))
    recall = np.sum(true_positives)/(np.sum(true_positives) + np.sum(false_negatives))
    f1_score = (2*precision * recall)/(precision + recall)
    return np.mean(abs_difference), np.sum(squared_difference)**0.5, false_negatives, false_positives, f1_score


threshold = 0.05
comparison_metrics(results1['val_matrix'], results1['p_matrix'], results2['val_matrix'],results2['p_matrix'], threshold)