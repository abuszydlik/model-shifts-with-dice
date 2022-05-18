import json
import numpy as np
import os


from carla import log
from carla.models.negative_instances import predict_negative_instances
from copy import deepcopy
from .dynamic_benchmark import DynamicBenchmark
from ..plotting.plot_dataset import calculate_boundary
from datetime import datetime


# TODO: What to do if a generator times out? do we accept different numbers of samples?
# TODO: Add benchmarks
class RecourseExperiment():
    """
    Allows to conduct an experiment about the dynamics of algorithmic recourse.

    Attributes:
        dataset (DataCatalog):
            Catalog containing a dataframe, set of train and test records, and the target.
        model (MLModelCatalog)
            Classifier with additional utilities required by CARLA.
        generators (List[RecourseGenerator]):
            List of one or more generators which will be measured in the experiment.
        experiment_name (str):
            Name of the experiment that will be used as part of the directory name where results are saved.
    """
    def __init__(self, dataset, model, generators, experiment_name='experiment', **kwargs):
        assert len(generators) != 0

        # Experiment data is saved into a new directory
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.experiment_name = f'{timestamp}_{experiment_name}'
        os.makedirs(f'../experiment_data/{self.experiment_name}')
        self.initial_dataset = deepcopy(dataset._df)
        self.initial_model = deepcopy(model)
        _, _, self.initial_boundary, self.x_min, self.x_max = calculate_boundary(self.initial_dataset,
                                                                                 self.initial_model)
        self.generators = generators

        # factuals are a list of instances that the model expects to belong to the negative class;
        # in order to acurately measure the performance of the dataset we never change the test set
        self.factuals = predict_negative_instances(model, dataset.df_train)
        self.factuals_index = self.factuals.index.tolist()

        pos_individuals = dataset.df_train.loc[dataset.df_train['target'] == dataset.positive]
        self.initial_pos_sample = pos_individuals.sample(n=min(len(pos_individuals), 100)).to_numpy()

        self.experiment_data = {}
        self.experiment_data['parameters'] = kwargs

        self.benchmarks = {}
        for g in self.generators:
            self.experiment_data[g.name] = {0: {}}
            self.benchmarks[g.name] = DynamicBenchmark(model, g.recourse_method, g, self.factuals)

    def run(self, total_recourse=0.8, recourse_per_epoch=0.01):
        """
        Driver code to execute an experiment that allows to compare the dynamics of recourse
        applied by some generator to a benchmark described by Wachter et al. (2017).

        Attributes:
            total_recourse (float):
                Value between 0 and 1 representing the proportion of samples from the training set
                which should have recourse applied to them throughout the experiment.
            recourse_per_epoch (float):
                Value between 0 and 1 representing the proportion of samples from the training set
                which should have recourse applied to them in a single iteration.
        """
        path = f'../experiment_data/{self.experiment_name}'

        # Convert ratio of samples that should undergo recourse in a single epoch into a number
        recourse_per_epoch = max(int(recourse_per_epoch * len(self.factuals)), 1)
        # Convert ratio of samples that should undergo recourse in total into a number of epochs
        epochs = max(int(min(total_recourse, 1) * len(self.factuals) / recourse_per_epoch), 1)

        for g in self.generators:
            self.benchmarks[g.name].start(self.experiment_data, path,
                                          self.initial_model, self.initial_pos_sample)

        for epoch in range(epochs - 1):
            log.info(f"Starting epoch {epoch + 1}")
            # Generate a subset S of factuals that have never been encountered by the generators
            sample_size = min(recourse_per_epoch, len(self.factuals_index))
            current_factuals_index = np.random.choice(self.factuals_index, replace=False, size=sample_size)
            # We do not want to accidentally generate a counterfactual from a counterfactual
            self.factuals_index = [f for f in self.factuals_index if f not in current_factuals_index]

            # Apply the same set of actions on all generators passed to the experiment
            for g in self.generators:
                self.benchmarks[g.name].next_iteration(self.experiment_data, path,
                                                       current_factuals_index,
                                                       self.initial_model,
                                                       self.initial_pos_sample,
                                                       self.initial_boundary,
                                                       self.x_min, self.x_max)

        # Measure the quality of recourse
        self.experiment_data['evaluation'] = {}
        for g in self.generators:
            log.info(self.benchmarks[g.name].compute_ynn())
            log.info(self.benchmarks[g.name].compute_distances())
            log.info(self.benchmarks[g.name].compute_constraint_violation())
            log.info(self.benchmarks[g.name].compute_redundancy())
            success_rate = g.num_found / max(len(self.factuals.index) - len(self.factuals_index), 1)
            self.experiment_data['evaluation'][g.name] = {'success_rate': success_rate}

    def save_data(self, path=None):
        """
        Write the data collected throughout the experiment into a .json file.

        Args:
            path (str):
                Directory where the dictionary of experiment data should be written.
        """

        path = path or f'../experiment_data/{self.experiment_name}/measurements.json'
        with open(path, 'w') as outfile:
            json.dump(self.experiment_data, outfile, sort_keys=True, indent=4)