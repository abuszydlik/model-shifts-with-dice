from .model import disagreement_distance, boundary_distance, model_MMD
from .distribution import measure_distribution, test_MMD
from .performance import measure_performance


def measure(generator, initial_model, initial_samples, initial_boundary=None, x_min=None, x_max=None):
    """
    Quantify the dataset and model and save into `experiment_data`.

    Args:
        generator (RecourseGenerator):
            Recourse generator along with utilities required to conduct experiments.
        epoch (int):
            Current epoch in the experiment.
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
    results['boundary_distance'] = boundary_distance(generator.dataset, generator.model)

    # Measure the MMD of the distribution and the model
    if initial_boundary is None:
        results['MMD'] = 0
        results['model_MMD'] = 0
    else:
        results['MMD'] = test_MMD(generator.dataset, initial_samples)
        results['model_MMD'] = model_MMD(generator.dataset, generator.model, initial_boundary, x_min, x_max)

    return results
