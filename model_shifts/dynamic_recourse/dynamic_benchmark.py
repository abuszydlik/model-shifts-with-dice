import pandas as pd
import timeit


from carla import MLModelCatalog
from carla.evaluation.benchmark import Benchmark
from copy import deepcopy
from ..metrics.measurement import measure
from ..plotting.plot_dataset import plot_distribution


class DynamicBenchmark(Benchmark):
    def __init__(self, mlmodel, recourse_method, generator, factuals):
        self._mlmodel = mlmodel
        self._recourse_method = recourse_method
        self._generator = generator
        self._factuals = deepcopy(factuals)
        self._counterfactuals = None
        self._epoch = 0
        self._timer = 0

        # Avoid using scaling and normalizing more than once
        if isinstance(mlmodel, MLModelCatalog):
            self._mlmodel.use_pipeline = False

    def start(self, experiment_data, path, initial_model, initial_pos_sample):
        experiment_data[self._generator.name][0] = measure(self._generator,
                                                           initial_model,
                                                           initial_pos_sample)

        # Plot initial data distributions
        plot_distribution(self._generator.dataset, self._generator.model, path,
                          self._generator.name, 'distribution', self._epoch)

        self._generator.update_model()

    def next_iteration(self, experiment_data, path, current_factuals_index,
                       initial_model, initial_pos_sample, initial_boundary, x_min, x_max):
        experiment_data[self._generator.name][self._epoch + 1] = {}

        # Find relevant factuals
        current_factuals = self._generator.dataset._df.iloc[current_factuals_index]

        # Apply recourse
        start_time = timeit.default_timer()
        counterfactuals = self._generator.apply(current_factuals)
        if self._counterfactuals is None:
            self._counterfactuals = counterfactuals
        else:
            self._counterfactuals = pd.concat([self._counterfactuals, counterfactuals], axis=0)

        self._timer += timeit.default_timer() - start_time
        self._generator.num_found += len(counterfactuals.index)
        self._generator.update_model()

        # Measure the data distribution and performance of the model
        experiment_data[self._generator.name][self._epoch + 1] = measure(self._generator,
                                                                         initial_model,
                                                                         initial_pos_sample,
                                                                         initial_boundary,
                                                                         x_min, x_max)

        # Plot data distributions
        plot_distribution(self._generator.dataset, self._generator.model, path,
                          self._generator.name, 'distribution', self._epoch + 1)

        # Re-create the generator on new model
        self._generator.update_generator()
        self._epoch += 1
