import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_samples(mean, cov, size, max_value):
    if isinstance(cov, np.ndarray):
        samples = np.random.multivariate_normal(mean, cov, size=size)
    else:
        distribution = np.random.normal(int(max_value / 2), np.sqrt(cov), size)
        counts, bins = np.histogram(distribution, np.linspace(0, max_value, num=(max_value + 1)))
        result = []
        for index, bin in enumerate(bins):
            values = [bin] * counts[index]
            result.extend(values)
        samples = np.array(result)
    return samples


def generate_continuous_dataset(means0, covs0, sizes0, means1, covs1, sizes1, file_name="data.csv"):
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
    # Generate the random samples
    samples0 = np.random.multivariate_normal(means0[0], covs0[0], sizes0[0])
    samples1 = np.random.multivariate_normal(means1[0], covs1[0], sizes1[0])

    for i, _ in enumerate(sizes0[1:]):
        new_samples = np.random.multivariate_normal(means0[i + 1], covs0[i + 1], sizes0[i + 1])
        samples0 = np.r_[samples0, new_samples]

    for i, _ in enumerate(sizes1[1:]):
        new_samples = np.random.multivariate_normal(means1[i + 1], covs1[i + 1], sizes1[i + 1])
        samples1 = np.r_[samples1, new_samples]

    # Append labels to the classes
    class0 = np.c_[samples0, np.zeros(len(samples0), dtype=np.int8)]
    class1 = np.c_[samples1, np.ones(len(samples1), dtype=np.int8)]
    colors = np.array(['#1f77b4', '#ff7f0e'])

    # Construct the dataset
    dataset = np.r_[class0, class1]

    # Plot the resulting distribution
    if dataset.shape[1] == 3:
        plt.scatter(dataset[:, 0], dataset[:, 1], c=colors[dataset[:, 2].astype(int)])
        plt.axis('equal')
        plt.grid(True)
        plt.show()

    # Store in a csv file
    dataframe = pd.DataFrame(data=dataset, columns=['feature1', 'feature2', 'target'])
    dataframe.to_csv(file_name, index=False, float_format='%1.4f')
    return dataset


def generate_categorical_samples(size, ranges):
    # Initialize the numpy array representing all samples in the class
    samples = np.zeros((size, len(ranges)), dtype=np.int)

    # Iterate through all features
    for i in range(len(ranges)):
        mean = ranges[i][0] + (ranges[i][1] - ranges[i][0]) / 2
        # We want to ensure that effectively all observations are in the provided range
        std = (ranges[i][1] - mean) / 3.5

        # Generate a distribution for this feature
        distribution = np.random.normal(mean, std, size)
        for j in range(len(distribution)):
            # Clamp to the given range
            value = sorted((ranges[i][0], int(distribution[j]), ranges[i][1]))[1]
            samples[j, i] = value

    return samples


def generate_categorical_dataset(size0, ranges0, size1, ranges1, file_name="data.csv"):

    samples0 = generate_categorical_samples(size0, ranges0)
    samples1 = generate_categorical_samples(size1, ranges1)

    # Append labels to the classes
    class0 = np.c_[samples0, np.zeros(len(samples0), dtype=np.int)]
    class1 = np.c_[samples1, np.ones(len(samples1), dtype=np.int)]
    colors = np.array(['#1f77b4', '#ff7f0e'])

    # Construct the dataset
    dataset = np.r_[class0, class1]

    # Plot the resulting distribution
    if dataset.shape[1] == 3:
        plt.scatter(dataset[:, 0], dataset[:, 1], c=colors[dataset[:, 2].astype(int)])
        plt.axis('equal')
        plt.grid(True)
        plt.show()

    # Store in a csv file
    columns = [f'feature{index + 1}' for index in range(len(ranges0))]
    columns.append('target')
    dataframe = pd.DataFrame(data=dataset, columns=columns)
    dataframe.to_csv(file_name, index=False, float_format='%1.4f')
    return dataset
