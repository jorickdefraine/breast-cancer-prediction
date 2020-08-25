import numpy as np


def rmsle(predicted_class, real_classe):
    """
    Root Mean Squared Logarithmic Error
    """

    somme = 0
    for i in range(len(real_classe)):
        somme += (np.log(predicted_class[i] + 1) - np.log(real_classe[i] + 1)) ** 2
    score = np.sqrt((1 / len(real_classe)) * somme)
    return score