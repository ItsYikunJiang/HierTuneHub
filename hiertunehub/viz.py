from typing import Optional
import os
import json
import logging

import numpy as np
import matplotlib.pyplot as plt


LOGGER = logging.getLogger(__name__)
CONSOLE_HANDLER = logging.StreamHandler()
LOGGER.addHandler(CONSOLE_HANDLER)
FORMATTER = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
CONSOLE_HANDLER.setFormatter(FORMATTER)


class ResultParser:
    def __init__(self, result_path: str, metric: Optional[str] = None):
        """
        This class is used to parse the multiple results from different runs.
        :param result_path: The path to the directory containing the results.
        :param metric: The metric to be visualized. If None, the results should be a float value. If not None,
        the results should be a dictionary containing the metric as the key.
        """
        self.result_path = result_path

        # read all the files in the directory and store the results
        self.result_matrix = list()
        self.min_p = np.inf

        for root, dirs, files in os.walk(result_path):
            for file in files:
                if file.endswith('.json'):
                    with open(os.path.join(root, file), 'r') as f:
                        run_result = json.load(f)
                        scores = np.zeros(len(run_result))
                        self.min_p = len(run_result) if len(run_result) < self.min_p else self.min_p
                        for i, r in enumerate(run_result):
                            scores[i] = r['result'] if metric is None else r['result'][metric]

                        self.result_matrix.append(scores)

        LOGGER.info(f"Found {len(self.result_matrix)} runs in {result_path}, with a minimum of {self.min_p} trials.")

        self.metric = metric
        self.min_result_matrix = np.zeros((len(self.result_matrix), self.min_p))
        for i, res in enumerate(self.result_matrix):
            scores = res[:self.min_p]
            best_so_far_score = np.zeros(len(scores))
            for j in range(len(scores)):
                best_so_far_score[j] = np.max(scores[:j + 1])

            self.min_result_matrix[i] = best_so_far_score

    def get_mean_sd(self):
        return np.mean(self.min_result_matrix, axis=0), np.std(self.min_result_matrix, axis=0)


class ResultVisualizer:
    def __init__(self,
                 result_paths: list[str],
                 metric: Optional[str] = None,
                 names: Optional[list[str]] = None,
                 ):
        """
        This class is used to visualize results from multiple runs with different configurations.
        :param result_paths: A list of paths to the directories containing the results.
        :param metric: The metric to be visualized. If None, the results should be a float value. If not None,
        the results should be a dictionary containing the metric as the key.
        :param names: The names of the different configurations.
        """

        self.result_parser = [ResultParser(result_path, metric) for result_path in result_paths]
        self.names = names if names is not None else [f'Config {i}' for i in range(len(result_paths))]

    def plot(self,
             title: str = 'Results',
             x_label: str = 'Number of Trials',
             y_label: str = 'Metric',
             color: Optional[list[str]] = None,
             ci: bool = True,
             ci_multiplier: float = 1.96,
             alpha: float = 0.2,
             figsize: tuple = (10, 5),
             plt_show: bool = True,
             save_path: Optional[str] = None
             ) -> plt.Axes:
        """
        Plot the results.
        :param title: The title of the plot.
        :param x_label: The x-axis label.
        :param y_label: The y-axis label.
        :param color: The color of the lines and the confidence interval.
        :param ci: Whether to plot the confidence interval.
        :param ci_multiplier: The multiplier to calculate the confidence interval with respect to the standard deviation.
        :param alpha: The transparency of the confidence interval.
        :param figsize: The size of the figure.
        :param plt_show: Whether to show the plot.
        :param save_path: The path to save the plot. If None, the plot will not be saved.
        """
        if color is None:
            if len(self.result_parser) > 10:
                raise ValueError("Too many configurations to plot without specifying colors.")
            color = ["C" + str(i) for i in range(len(self.result_parser))]

        fig, ax = plt.subplots(figsize=figsize)

        for i, result_parser in enumerate(self.result_parser):
            mean, sd = result_parser.get_mean_sd()
            ax.plot(mean, label=self.names[i], color=color[i] if color is not None else None)

            if ci:
                ax.fill_between(range(len(mean)),
                                mean - ci_multiplier * sd,
                                mean + ci_multiplier * sd,
                                alpha=alpha,
                                color=color[i] if color is not None else None)

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend()

        if plt_show:
            plt.show()

        if save_path is not None:
            fig.savefig(save_path)

        return ax
