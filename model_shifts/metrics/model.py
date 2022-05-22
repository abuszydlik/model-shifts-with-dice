import numpy as np


from .distribution import MMD, MMD_null_hypothesis


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


def class_decisiveness(data, target, model, cls):
    """Calculates the pseudo-distance of points to the decision boundary
    measured as the average probability of classification centered around 0.5.
    High value corresponds to a large margin of classification.

    Args:
        data (pandas.DataFrame):
            Records along with their labels.
        target (str):
            Name of the column that contains the target variable.
        model (MLModelCatalog):
            Classifier with additional utilities required by CARLA.
        cls (str):
            Encoding of the class to measure (positive or negative).

    Returns:
        float: Pseudo-distance from decision boundary
    """
    samples = data.loc[data[target] == cls]
    proba = model.predict_proba(samples.loc[:, samples.columns != target])
    return np.linalg.norm(proba[:, cls] - 0.5) / len(proba)


def decisiveness(dataset, model):
    """Measures the boundary pseudo-distance for both classes.

    Args:
        dataset (DataCatalog):
            Catalog containing a dataframe, set of train and test records, and the target.
        model (MLModelCatalog):
            Classifier with additional utilities required by CARLA.

    Returns:
        dict: A dictionary with values calculated for both classes.
    """
    return {
        'positive': class_decisiveness(dataset._df_test, dataset.target, model, dataset.positive),
        'negative': class_decisiveness(dataset._df_test, dataset.target, model, dataset.negative)
    }


def model_MMD(dataset, model, initial_proba, calculate_p):
    """Calculates the MMD on the probabilities of classification assigned by the model
    to the set of (all)  instances. Allows to quantify the model shift.

    Args:
        dataset (DataCatalog):
            Catalog containing a dataframe, set of train and test records, and the target.
        model (MLModelCatalog):
            Classifier with additional utilities required by CARLA.
        initial_proba (numpy.ndarray):
            Probabilities assigned by the model to a set of samples.

    Returns:
        dict: A dictionary with an estimate of current MMD and p-value for that estimate.
    """
    sample = np.random.randint(2, size=min(len(initial_proba), 10000)).astype('bool')
    updated_proba = model.predict_proba(dataset._df.loc[:, dataset._df.columns != dataset.target])
    mmd = MMD(initial_proba[sample], updated_proba[sample])

    result = {'value': mmd}

    if calculate_p:
        iterations = 1000
        mmd_null = MMD_null_hypothesis(initial_proba, updated_proba, iterations)
        result['p'] = max(1 / iterations, np.count_nonzero(mmd_null >= mmd) / iterations)

    return result
