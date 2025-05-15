from typing import Optional
import os
import json
import logging

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

LOGGER = logging.getLogger(__name__)
CONSOLE_HANDLER = logging.StreamHandler()
LOGGER.addHandler(CONSOLE_HANDLER)
FORMATTER = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
CONSOLE_HANDLER.setFormatter(FORMATTER)
LOGGER.setLevel(logging.INFO)


class ResultParser:
    def __init__(self,
                 result_path: str,
                 metric: Optional[str] = None,
                 mode: str = 'min',
                 cutoff: Optional[int] = None):
        """
        This class is used to parse the multiple results from different runs.
        Results are stored in result_matrix with shape (num_runs, num_trials).
        :param result_path: The path to the directory containing the results.
        :param metric: The metric to be visualized. If None, the results should be a float value. If not None,
        the results should be a dictionary containing the metric as the key.
        :param mode: The mode to be used for the metric. Can be 'min' or 'max'.
        :param cutoff: The number of trials to be used for the metric. If None, all trials will be used.
        """
        self.result_path = result_path
        self.mode = mode

        # read all the files in the directory and store the results
        self.result_matrix = list()
        self.elapsed_time = list()
        if cutoff is not None:
            self.min_p = cutoff
        else:
            self.min_p = np.inf

        for root, dirs, files in os.walk(result_path):
            for file in files:
                if file.endswith('.json'):
                    with open(os.path.join(root, file), 'r') as f:
                        run_result = json.load(f)
                        scores = np.zeros(len(run_result))
                        elapsed_time = np.zeros(len(run_result))
                        if cutoff is None:
                            self.min_p = len(run_result) if len(run_result) < self.min_p else self.min_p
                        for i, r in enumerate(run_result):
                            scores[i] = r['result'] if metric is None else r['result'][metric]
                            elapsed_time[i] = r['result']['elapsed_time']
                            if scores[i] == float('-inf') or scores[i] == float('inf'):
                                scores[i] = np.inf if mode == 'min' else -np.inf

                        self.result_matrix.append(scores)
                        self.elapsed_time.append(elapsed_time)

        LOGGER.info(f"Found {len(self.result_matrix)} runs in {result_path}, with a minimum of {self.min_p} trials.")

        self.metric = metric
        self.min_result_matrix = np.zeros((len(self.result_matrix), self.min_p))
        self.elapsed_time_matrix = np.zeros((len(self.result_matrix), self.min_p))
        for i, res in enumerate(self.result_matrix):
            scores = res[:self.min_p]
            best_so_far_score = np.zeros(len(scores))
            for j in range(len(scores)):
                best_so_far_score[j] = np.max(scores[:j + 1]) if mode == 'max' else np.min(scores[:j + 1])

            self.min_result_matrix[i] = best_so_far_score
            self.elapsed_time_matrix[i] = self.elapsed_time[i][:self.min_p]

    def get_mean_sd(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the mean and standard deviation of the results across the runs.
        :return: A tuple containing the mean and standard deviation of the results with shape (num_runs,).
        """
        return np.mean(self.min_result_matrix, axis=0), np.std(self.min_result_matrix, axis=0)

    def get_best_when(self) -> np.ndarray:
        """
        Get the number of trials when the best result was achieved.
        :return: An array shape of (num_runs, ) containing the number of trials when the best result was achieved.
        """
        if self.mode == 'max':
            return np.argmax(self.min_result_matrix, axis=1) + 1
        else:
            return np.argmin(self.min_result_matrix, axis=1) + 1

    def get_best_results(self) -> np.ndarray:
        """
        Get the best results achieved.
        :return: An array shape of (num_runs, ) containing the best results achieved.
        """
        if self.mode == 'max':
            return np.max(self.min_result_matrix, axis=1)
        else:
            return np.min(self.min_result_matrix, axis=1)

    def get_best_elapsed_time(self) -> np.ndarray:
        """
        Get the total elapsed time for the best result achieved.
        """
        if self.mode == 'max':
            best_idx = np.argmax(self.min_result_matrix, axis=1)
        else:
            best_idx = np.argmin(self.min_result_matrix, axis=1)

        best_elapsed_time = np.zeros(len(self.result_matrix))
        for i, idx in enumerate(best_idx):
            best_elapsed_time[i] = np.sum(self.elapsed_time_matrix[i][:idx + 1])

        return best_elapsed_time


class ResultVisualizer:
    def __init__(self,
                 result_paths: list[str],
                 metric: Optional[str] = None,
                 mode: str = 'min',
                 names: Optional[list[str]] = None,
                 cutoff: Optional[int] = None
                 ):
        """
        This class is used to visualize results from multiple runs with different configurations.
        :param result_paths: A list of paths to the directories containing the results.
        :param metric: The metric to be visualized. If None, the results should be a float value. If not None,
        the results should be a dictionary containing the metric as the key.
        :param names: The names of the different configurations.
        :param mode: The mode to be used for the metric. Can be 'min' or 'max'.
        :param cutoff: The number of trials to be used for the metric. If None, all trials will be used.
        """

        self.result_parser = [ResultParser(result_path, metric, mode, cutoff) for result_path in result_paths]
        self.names = names if names is not None else [f'Config {i}' for i in range(len(result_paths))]

    def _get_color_palette(self, color: Optional[list]) -> list:
        if color is None:
            if len(self.result_parser) > 10:
                raise ValueError("Too many configurations to plot without specifying colors.")
            color = ["C" + str(i) for i in range(len(self.result_parser))]

        return color

    def plot_metric_score(self,
                          title: str = 'Results',
                          x_label: str = 'Number of Trials',
                          y_label: str = 'Metric',
                          color: Optional[list[str]] = None,
                          cutoff: Optional[int] = None,
                          y_lim: Optional[tuple] = None,
                          ci: bool = True,
                          ci_multiplier: float = 1.96,
                          alpha: float = 0.2,
                          figsize: tuple = (10, 5),
                          plt_show: bool = True,
                          save_path: Optional[str] = None,
                          ax: Optional[plt.Axes] = None
                          ) -> plt.Axes:
        """
        Plot the best metric score based on the number of trials.
        :param title: The title of the plot.
        :param x_label: The x-axis label.
        :param y_label: The y-axis label.
        :param color: The color of the lines and the confidence interval.
        :param cutoff: The number of trials to plot. If None, all trials will be plotted.
        :param y_lim: The limits of the y-axis.
        :param ci: Whether to plot the confidence interval.
        :param ci_multiplier: The multiplier to calculate the confidence interval with respect to the standard deviation.
        :param alpha: The transparency of the confidence interval.
        :param figsize: The size of the figure.
        :param plt_show: Whether to show the plot.
        :param save_path: The path to save the plot. If None, the plot will not be saved.
        :param ax: The axes to plot on. If None, a new figure will be created.
        """
        color = self._get_color_palette(color)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        for i, result_parser in enumerate(self.result_parser):
            mean, sd = result_parser.get_mean_sd()
            if cutoff is not None:
                mean = mean[:cutoff]
                sd = sd[:cutoff]
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
        if y_lim is not None:
            ax.set_ylim(y_lim)
        ax.legend()

        if plt_show:
            plt.show()

        if save_path is not None:
            plt.savefig(save_path)

        return ax

    def plot_all_best_when(self,
                           title: str = 'Number of Trials When Best Result was Achieved',
                           x_label: str = 'Number of Trials',
                           y_label: str = 'Frequency',
                           color: Optional[list] = None,
                           kde_only: bool = True,
                           figsize: tuple = (10, 5),
                           plt_show: bool = False,
                           save_path: Optional[str] = None,
                           ax: Optional[plt.Axes] = None
                           ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        color = self._get_color_palette(color)

        for i, name in enumerate(self.names):
            best_when = self.result_parser[i].get_best_when()

            self._plot_hist(data=best_when,
                            title=title,
                            name=name,
                            x_label=x_label,
                            y_label=y_label,
                            color=color[i],
                            kde_only=kde_only,
                            figsize=figsize,
                            plt_show=plt_show,
                            save_path=save_path,
                            ax=ax)

        ax.legend()

        return ax

    def plot_best_elapsed_time(self,
                               name: str,
                               title: str = 'Best Elapsed Time',
                               x_label: str = 'Elapsed Time',
                               y_label: str = 'Frequency',
                               color: str = 'C0',
                               kde_only: bool = False,
                               figsize: tuple = (10, 5),
                               plt_show: bool = True,
                               save_path: Optional[str] = None,
                               ax: Optional[plt.Axes] = None
                               ) -> plt.Axes:
        best_elapsed_time = self.result_parser[self.names.index(name)].get_best_elapsed_time()

        return self._plot_hist(data=best_elapsed_time,
                               title=f"{title} for {name}",
                               name=name,
                               x_label=x_label,
                               y_label=y_label,
                               color=color,
                               kde_only=kde_only,
                               figsize=figsize,
                               plt_show=plt_show,
                               save_path=save_path,
                               ax=ax)

    def plot_best_elapsed_time_all(self,
                                   title: str = 'Best Elapsed Time',
                                   x_label: str = 'Elapsed Time',
                                   y_label: str = 'Frequency',
                                   color: Optional[list] = None,
                                   kde_only: bool = True,
                                   figsize: tuple = (10, 5),
                                   plt_show: bool = False,
                                   save_path: Optional[str] = None,
                                   ax: Optional[plt.Axes] = None
                                   ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        color = self._get_color_palette(color)

        for i, name in enumerate(self.names):
            best_elapsed_time = self.result_parser[i].get_best_elapsed_time()

            self._plot_hist(data=best_elapsed_time,
                            title=title,
                            name=name,
                            x_label=x_label,
                            y_label=y_label,
                            color=color[i],
                            kde_only=kde_only,
                            figsize=figsize,
                            plt_show=plt_show,
                            save_path=save_path,
                            ax=ax)

        ax.legend()

        return ax

    def plot_best_when(self,
                       name: str,
                       title: str = 'Number of Trials When Best Result was Achieved',
                       x_label: str = 'Number of Trials',
                       y_label: str = 'Frequency',
                       color: str = 'C0',
                       kde_only: bool = False,
                       figsize: tuple = (10, 5),
                       plt_show: bool = True,
                       save_path: Optional[str] = None,
                       ax: Optional[plt.Axes] = None
                       ) -> plt.Axes:
        """
        Plot the number of trials when the best result was achieved.
        :param name: The name of the configuration to plot.
        :param title: The title of the plot.
        :param x_label: The x-axis label.
        :param y_label: The y-axis label.
        :param color: The color of the plot.
        :param kde_only: Whether to plot only the kernel density estimate.
        :param figsize: The size of the figure.
        :param plt_show: Whether to show the plot.
        :param save_path: The path to save the plot. If None, the plot will not be saved.
        :param ax: The axes to plot on. If None, a new figure will be created.
        """
        best_when = self.result_parser[self.names.index(name)].get_best_when()

        return self._plot_hist(data=best_when,
                               title=f"{title} for {name}",
                               name=name,
                               x_label=x_label,
                               y_label=y_label,
                               color=color,
                               kde_only=kde_only,
                               figsize=figsize,
                               plt_show=plt_show,
                               save_path=save_path,
                               ax=ax)

    def plot_all_best_result(self,
                             title: str = 'Best Result Achieved',
                             x_label: str = 'Energy Distance',
                             y_label: str = 'Frequency',
                             color: Optional[list] = None,
                             kde_only: bool = True,
                             figsize: tuple = (10, 5),
                             plt_show: bool = False,
                             save_path: Optional[str] = None,
                             ax: Optional[plt.Axes] = None
                             ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        color = self._get_color_palette(color)

        for i, name in enumerate(self.names):
            best_result = self.result_parser[i].get_best_results()

            self._plot_hist(data=best_result,
                            title=title,
                            name=name,
                            x_label=x_label,
                            y_label=y_label,
                            color=color[i],
                            kde_only=kde_only,
                            figsize=figsize,
                            plt_show=plt_show,
                            save_path=save_path,
                            ax=ax)
        ax.legend()

        return ax

    def plot_best_result(self,
                         name: str,
                         title: str = 'Best Result Achieved',
                         x_label: str = 'Energy Distance',
                         y_label: str = 'Frequency',
                         color: str = 'C0',
                         kde_only: bool = False,
                         figsize: tuple = (10, 5),
                         plt_show: bool = True,
                         save_path: Optional[str] = None,
                         ax: Optional[plt.Axes] = None
                         ) -> plt.Axes:
        """
        Plot the best result achieved for a configuration.
        :param name: The name of the configuration to plot.
        :param title: The title of the plot.
        :param x_label: The x-axis label.
        :param y_label: The y-axis label.
        :param color: The color of the plot.
        :param kde_only: Whether to plot only the kernel density estimate.
        :param figsize: The size of the figure.
        :param plt_show: Whether to show the plot.
        :param save_path: The path to save the plot. If None, the plot will not be saved.
        :param ax: The axes to plot on. If None, a new figure will be created.
        """
        best_result = self.result_parser[self.names.index(name)].get_best_results()

        return self._plot_hist(data=best_result,
                               title=f"{title} for {name}",
                               name=name,
                               x_label=x_label,
                               y_label=y_label,
                               color=color,
                               kde_only=kde_only,
                               figsize=figsize,
                               plt_show=plt_show,
                               save_path=save_path,
                               ax=ax)

    @staticmethod
    def _plot_hist(data: np.ndarray,
                   title: str,
                   name: str,
                   x_label: str,
                   y_label: str,
                   color: str,
                   kde_only: bool,
                   figsize: tuple,
                   plt_show: bool,
                   save_path: Optional[str],
                   ax: Optional[plt.Axes]
                   ) -> plt.Axes:
        """
        Plot the histogram of the data.
        :param data: The data to plot.
        :param title: The title of the plot.
        :param x_label: The x-axis label.
        :param y_label: The y-axis label.
        :param color: The color of the plot.
        :param kde_only: Whether to plot only the kernel density estimate.
        :param figsize: The size of the figure.
        :param plt_show: Whether to show the plot.
        :param save_path: The path to save the plot. If None, the plot will not be saved.
        :param ax: The axes to plot on. If None, a new figure will be created.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if kde_only:
            sns.kdeplot(
                data,
                ax=ax,
                color=color,
                label=name
            )
        else:
            sns.histplot(
                data,
                ax=ax,
                color=color,
                kde=True,
                label=name
            )

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        if plt_show:
            plt.show()

        if save_path is not None:
            plt.savefig(save_path)

        return ax
