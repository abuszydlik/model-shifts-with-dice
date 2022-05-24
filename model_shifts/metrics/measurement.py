from .model import disagreement_distance, decisiveness, model_MMD
from .distribution import measure_distribution, test_MMD
from .performance import measure_performance


def measure(generator, initial_model, initial_samples, initial_proba, calculate_p):
    """
    Quantify the dataset and model and save into `experiment_data`.

    Args:
        generator (RecourseGenerator):
            Recourse generator along with utilities required to conduct experiments.
        initial_model (MLModelCatalog):
            Copy of the classifier before the implementation of recourse.
        initial_samples (dict of numpy.ndarray):
            Samples from the positive and negative class before the implementation of recourse.
        initial_proba (numpy.ndarray):
            Predicted probabilities assigned to samples before the implementation of recourse.
        calculate_p (Boolean):
            If True, the statistical significance is calculated for MMD of distribution and model.

    Returns:
        dict: A dictionary storing all measurements for the current epoch.
    """
    results = {}

    # Measure the distributions of data
    results['distribution'] = measure_distribution(generator.dataset)

    # Measure the current performance of models
    results['performance'] = measure_performance(generator.dataset, generator.model)

    # Measure the disagreement between current model and the initial model
    results['disagreement'] = disagreement_distance(generator.dataset._df_test, generator.dataset.target,
                                                    initial_model, generator.model)

    # Measure the average distance of a sample from the decision boundary
    results['decisiveness'] = decisiveness(generator.dataset, generator.model)

    # Measure the MMD of the distribution and the model
    results['MMD'] = test_MMD(generator.dataset, initial_samples, calculate_p)
    results['model_MMD'] = model_MMD(generator.dataset, generator.model, initial_proba, calculate_p)

    return results
