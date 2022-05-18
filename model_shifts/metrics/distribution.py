import numpy as np

from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.gaussian_process.kernels import RBF


def MMD(x, y, scale=0.5):
    """
    Calculates the Maximum Mean Discrepancy (MMD) between sets of samples from two probability distribution.
    MMD = ||µp - µq||^{2}_{H} where µp, µq are mean embeddings in H, a Reproducing Kernel Hilbert Space.
    This can be used either per class (which may be more suitable to assess shifts) or on the whole dataset.

    Args:
        x (numpy.ndarray):
            An n-dimensional sample of length `l` from the first distribution.
        y (numpy.ndarray):
            An n-dimensional sample of length `l` from the second distribution.
        scale (float):
            Length-scale of the kernel, influences the smoothness of the transformation.

    Returns:
        float: Measure of distance between two distributions (0 means distributions are the same).
    """

    # RBF is the Gaussian kernel allowing to embed the distribution in RKHS
    # As e^x = 1 + x + (1 / 2!) * x^2 ... we can capture all moments of the distribution
    k = RBF(length_scale=scale)

    # Implementation of Equation (3) from https://arxiv.org/pdf/1810.11953.pdf
    m, n = len(x), len(y)

    # We apply the kernel trick to get an unbiased empirical estimate of the squared population MMD
    Kxx = 1 / (m ** 2 - m) * (np.sum(k(x, x)) - np.trace(k(x, x)))
    Kxy = 2 / (m * n) * np.sum(k(x, y))
    Kyy = 1 / (n ** 2 - n) * (np.sum(k(y, y)) - np.trace(k(y, y)))

    return np.sqrt(np.abs(Kxx + Kyy - Kxy))


def MMD_null_hypothesis(x, y, iterations=10000):
    """
    Calculates the MMD for a set of permutations of samples from the two distributions
    to measure whether the shift should be considered significant. This works under the assumption
    that if samples `x` and `y` come from the same distribution (under the null hypothesis),
    then the MMD of permutations of these samples should be similar to MMD(x, y).

    Args:
        x (numpy.ndarray):
            An n-dimensional sample of length `l` from the first distribution.
        y (numpy.ndarray):
            An n-dimensional sample of length `l` from the second distribution.
        iterations (int):
            Number of permutations that should be created for the testing.

    Returns:
        numpy.ndarray: Array containing MMDs of all permutations of `x` and `y`.
    """

    n = len(x)
    mmd_null = np.zeros(iterations)
    for index in range(iterations):
        permutation = np.random.permutation(np.r_[x, y])
        mmd_null[index] = MMD(permutation[:n], permutation[n:])

    return mmd_null


def current_MMD(data, positive_class, initial_pos):
    """
    Measure the MMD between the initial distribution and the current distribution for both classes.

    Args:
        data (pandas.DataFrame):
            Records along with their labels.
        positive_class (int):
            Encoding of the positive class in the dataset.
        initial_pos (numpy.ndarray):
            A sample of points of the positive class from the initial distribution.

    Returns:
        dict: A dictionary containing current values of MMD for the positive and the negative class.
    """

    # Find individuals of both classes
    pos_individuals = data.loc[data['target'] == positive_class]

    # Sample individuals of the positive class, distribution of the negative class does not change over time
    # TODO this will fail if 100 individuals are not left
    updated_pos = pos_individuals.sample(n=min(len(pos_individuals), 100)).to_numpy()
    mmd_pos = MMD(initial_pos, updated_pos)

    iterations = 1000
    mmd_null = MMD_null_hypothesis(initial_pos, updated_pos, iterations)

    p_pos = max(1 / iterations, np.count_nonzero(mmd_null >= mmd_pos) / iterations)

    return {
        'value': mmd_pos,
        'p': p_pos
    }


def k_means(data, min_clusters=1, max_clusters=10):
    """
    Applies the k-means algorithm and automatically estimates the elbow point.
    The algorithm used to calculate the elbow point is described in 10.1109/ICDCSW.2011.20

    Args:
        data (pandas.DataFrame):
            Records along with their labels.
        min_clusters (int):
            Minimal number of clusters that is expected in the dataset.
        max_clusters (int):
            Maximal number of clusters that is expected in the dataset.

    Returns:
        int: Estimated number of clusters that yields the elbow point on an inertia graph.
    """
    clusters = []
    scores = []
    # Fit different potential numbers of clusters
    for k in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
        clusters.append(k)
        scores.append(kmeans.inertia_)

    # Automatically find the elbow point, this should change at some point during the application of AR
    # if the counterfactual instances form their own cluster(s), the value returned by this method should change.
    kneedle = KneeLocator(clusters, scores, curve="convex", direction="decreasing")

    return int(kneedle.elbow)


def class_statistics(dataset, aggregate):
    """
    Applies an aggregation function for the two classes described in the dataset.

    Args:
        dataset (DataCatalog):
            Catalog containing a dataframe, set of train and test records, and the target.
        aggregate (Callable):
            An aggregation function which can be applied on data.

    Returns:
        dict: Values return by the aggregation function applied on the positive and negative class.
    """
    features = dataset._df.loc[:, dataset._df.columns != dataset._target]
    positive_samples = features.loc[dataset._df[dataset._target] == dataset._positive]
    negative_samples = features.loc[dataset._df[dataset._target] == dataset._negative]

    return {
        "positive": aggregate(positive_samples).to_dict(),
        "negative": aggregate(negative_samples).to_dict()
    }


def measure_distribution(dataset):
    """
    Computes a set of statistical measures for the distribution of a dataset.

    Args:
        dataset (DataCatalog):
            Catalog containing a dataframe, set of train and test records, and the target.

    Returns:
        dict: Statistics calculated for a specific distribution of data.
    """
    num_clusters = k_means(dataset.df.loc[:, dataset.df.columns != dataset.target].to_numpy())
    means = class_statistics(dataset, np.mean)
    stds = class_statistics(dataset, np.std)

    return {
        "num_clusters": num_clusters,
        "means": means,
        "stds": stds
    }
