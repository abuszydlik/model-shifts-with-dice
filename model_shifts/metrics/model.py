import numpy as np


from ..plotting.plot_dataset import calculate_boundary
from .distribution import MMD


def disagreement_distance(data, target, initial_model, updated_model):
    """
    Calculates the Disagreement pseudo-distance defined in https://doi.org/10.1145/1273496.1273541
    as Pr(h(x) != h'(x)), that is the probability that labels assigned by one classifier do not agree
    with the labels assigned by another classifier. Simply put, it measures the overlap between models.
    As this is an empirical measure, we can vary the number of records in `data`.

    Args:
        data (pandas.DataFrame):
            A withheld set of records that should be predicted by the models (test set).
        target (str):
            The target column in the dataset.
        initial_model (MLModelCatalog):
            A model which was trained before recourse has been applied.
        updated_model (MLModelCatalog):
            A model retrained on a dataset with induced recourse.

    Returns:
        float: Probability that the two classifiers disagree on the label of a sample.
    """

    # Check how the initial model would assign labels to the test set
    initial_pred = np.argmax(initial_model.predict_proba(data.loc[:, data.columns != target]), axis=1)

    # Check how the updated model would assign labels to the test set
    updated_pred = np.argmax(updated_model.predict_proba(data.loc[:, data.columns != target]), axis=1)

    count_mismatch = 0
    for index, prediction in enumerate(initial_pred):
        if updated_pred[index] != prediction:
            count_mismatch += 1

    # Find the disagreement pseudo-distance
    return count_mismatch / len(initial_pred)


def boundary_distance(dataset, model):
    data = dataset._df_test
    positive = data.loc[data[dataset.target] == dataset.positive]
    negative = data.loc[data[dataset.target] == dataset.negative]

    positive_proba = model.predict_proba(positive.loc[:, positive.columns != dataset.target])
    negative_proba = model.predict_proba(negative.loc[:, negative.columns != dataset.target])

    return {
        'positive': np.linalg.norm(positive_proba[:, 0] - 0.5) / len(positive_proba),
        'negative': np.linalg.norm(negative_proba[:, 0] - 0.5) / len(negative_proba)
    }


def model_MMD(dataset, model, initial_boundary, x_min=None, x_max=None):
    _, _, z, _, _ = calculate_boundary(dataset._df, model, x_min=x_min, x_max=x_max)
    return MMD(initial_boundary, z)
