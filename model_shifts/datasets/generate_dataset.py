import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_dataset(means0, covs0, means1, covs1, num_samples=500, modality=[1.0], file_name="data.csv"):
    """
    Generates a dataset which contains `num_samples` samples divided into two normally-distributed classes.

    Args:
        means0 (list of numpy.ndarray): the means of all peaks in the distribution of class A.
        covs0 (list of numpy.ndarray): the covariance matrices for all peaks in the distribution of class A.
        means1 (list of numpy.ndarray): the means of all peaks in the distribution of class B.
        covs1 (list of numpy.ndarray): the covariance matrices for all peaks in the distribution of class B.
        num_samples (int): the total number of samples expected in the dataset.
        modality (list of float): the relative size of the peaks if distributions are multi-modal.
        file_name (str): the name of the file where the resulting dataset will be stored.

    Returns:
        numpy.ndarray: dataset containing samples of two classes distributed according to the (multi-modal) Gaussian.
    """

    # Validate dataset characteristics
    assert len(modality) == len(means0) == len(covs0) == len(means1) == len(covs1)
    assert sum(modality) == 1

    # Generate the random samples
    samples0 = np.random.multivariate_normal(means0[0], covs0[0], size=int((num_samples / 2) * modality[0]))
    samples1 = np.random.multivariate_normal(means1[0], covs1[0], size=int((num_samples / 2) * modality[0]))

    for index, mod in enumerate(modality[1:]):
        samples0 = np.r_[samples0, np.random.multivariate_normal(
            means0[index + 1], covs0[index + 1], size=int((num_samples / 2) * mod))]
        samples1 = np.r_[samples1, np.random.multivariate_normal(
            means1[index + 1], covs1[index + 1], size=int((num_samples / 2) * mod))]

    # Append labels to the classes
    class0 = np.c_[samples0, np.zeros(len(samples0), dtype=np.int8)]
    class1 = np.c_[samples1, np.ones(len(samples1), dtype=np.int8)]
    colors = np.array(['#1f77b4', '#ff7f0e'])

    # Construct the dataset
    dataset = np.r_[class0, class1]

    # Plot the resulting distribution
    plt.scatter(dataset[:, 0], dataset[:, 1], c=colors[dataset[:, 2].astype(int)])
    plt.axis('equal')
    plt.grid(True)
    plt.show()

    # Store in a csv file
    dataframe = pd.DataFrame(data=dataset, columns=['feature1', 'feature2', 'target'])
    dataframe.to_csv(file_name, index=False, float_format='%1.4f')
    return dataset
