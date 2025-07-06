from typing import Optional
import os
import json
import logging

import numpy as np
import pandas as pd
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

        self.early_stopping_idx = np.zeros(len(self.result_matrix), dtype=int)

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

    def early_stopping_criteria(self, k: int = 10, warmup: int = 50) -> np.ndarray:
        """
        Returns the vector of the number of trials when the early stopping criteria was met.
        :param k: After k trials, if the results do not improve, the early stopping criteria is met.
        :param warmup: The number of trials to warm up before applying the early stopping criteria.
        """
        for i, res in enumerate(self.min_result_matrix):
            if self.mode == 'max':
                best_so_far_idx = np.argmax(res[:warmup + k])
                for j in range(warmup + k, len(res)):
                    if res[j] > res[best_so_far_idx]:
                        best_so_far_idx = j
                    if j - best_so_far_idx >= k:
                        self.early_stopping_idx[i] = j + 1
                        break
            else:
                best_so_far_idx = np.argmin(res[:warmup + k])
                for j in range(warmup + k, len(res)):
                    if res[j] < res[best_so_far_idx]:
                        best_so_far_idx = j
                    if j - best_so_far_idx >= k:
                        self.early_stopping_idx[i] = j + 1
                        break

            if self.early_stopping_idx[i] == 0:
                self.early_stopping_idx[i] = len(res)

        return self.early_stopping_idx

    def get_early_stopping_mean_sd(self, k: int = 10, warmup: int = 50) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the mean and standard deviation of the results when the early stopping criteria was met.
        :param k: After k trials, if the results do not improve, the early stopping criteria is met.
        :param warmup: The number of trials to warm up before applying the early stopping criteria.
        :return: A tuple containing the mean and standard deviation of the results with shape (num_runs,).
        """
        early_stopping_idx = self.early_stopping_criteria(k=k, warmup=warmup)
        early_stopping_results = self.min_result_matrix.copy()

        for i in range(len(self.result_matrix)):
            early_stopping_results[i][early_stopping_idx[i]:] = np.nan

        return np.nanmean(early_stopping_results, axis=0), np.nanstd(early_stopping_results, axis=0)


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

    def _plot_all_with_data(self,
                            data: pd.DataFrame,
                            target: str,
                            title: str,
                            x_label: str,
                            y_label: str,
                            color: Optional[list] = None,
                            kde_only: bool = True,
                            figsize: tuple = (10, 5),
                            plt_show: bool = False,
                            save_path: Optional[str] = None,
                            ax: Optional[plt.Axes] = None,
                            **kwargs
                            ):

        color = self._get_color_palette(color)

        if kde_only:
            for i, name in enumerate(self.names):
                subset = data[data['name'] == name][target]
                self._plot_hist(data=subset,
                                title=title,
                                name=name,
                                x_label=x_label,
                                y_label=y_label,
                                color=color[i],
                                figsize=figsize,
                                plt_show=plt_show,
                                save_path=save_path,
                                ax=ax,
                                **kwargs)
            ax.legend()
        else:
            self._plot_non_kde_hist(
                data=data,
                x=target,
                title=title,
                x_label=x_label,
                y_label=y_label,
                figsize=figsize,
                plt_show=plt_show,
                save_path=save_path,
                ax=ax,
                **kwargs
            )

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
                           ax: Optional[plt.Axes] = None,
                           **kwargs
                           ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        color = self._get_color_palette(color)

        best_whens = pd.DataFrame(columns=['name', 'best_when'])

        for i, name in enumerate(self.names):
            best_when = self.result_parser[i].get_best_when()
            best_whens = pd.concat(
                [best_whens, pd.DataFrame({'name': [name] * len(best_when), 'best_when': best_when})])

        self._plot_all_with_data(
            data=best_whens,
            target='best_when',
            title=title,
            x_label=x_label,
            y_label=y_label,
            color=color,
            kde_only=kde_only,
            figsize=figsize,
            plt_show=plt_show,
            save_path=save_path,
            ax=ax,
            **kwargs
        )

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
                               ax: Optional[plt.Axes] = None,
                               **kwargs
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
                               ax=ax,
                               **kwargs)

    def plot_best_elapsed_time_all(self,
                                   title: str = 'Best Elapsed Time',
                                   x_label: str = 'Elapsed Time',
                                   y_label: str = 'Frequency',
                                   color: Optional[list] = None,
                                   kde_only: bool = True,
                                   figsize: tuple = (10, 5),
                                   plt_show: bool = False,
                                   save_path: Optional[str] = None,
                                   ax: Optional[plt.Axes] = None,
                                   **kwargs
                                   ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        color = self._get_color_palette(color)

        best_elapsed_times = pd.DataFrame(columns=['name', 'best_elapsed_time'])

        for i, name in enumerate(self.names):
            best_elapsed_time = self.result_parser[i].get_best_elapsed_time()
            best_elapsed_times = pd.concat(
                [best_elapsed_times, pd.DataFrame({'name': [name] * len(best_elapsed_time),
                                                   'best_elapsed_time': best_elapsed_time})])

        self._plot_all_with_data(
            data=best_elapsed_times,
            target='best_elapsed_time',
            title=title,
            x_label=x_label,
            y_label=y_label,
            color=color,
            kde_only=kde_only,
            figsize=figsize,
            plt_show=plt_show,
            save_path=save_path,
            ax=ax,
            **kwargs
        )

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
                       ax: Optional[plt.Axes] = None,
                       **kwargs
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
                               ax=ax,
                               **kwargs)

    def plot_all_best_result(self,
                             title: str = 'Best Result Achieved',
                             x_label: str = 'Energy Distance',
                             y_label: str = 'Frequency',
                             color: Optional[list] = None,
                             kde_only: bool = True,
                             figsize: tuple = (10, 5),
                             plt_show: bool = False,
                             save_path: Optional[str] = None,
                             ax: Optional[plt.Axes] = None,
                             **kwargs
                             ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        color = self._get_color_palette(color)

        best_results = pd.DataFrame(columns=['name', 'best_result'])

        for i, name in enumerate(self.names):
            best_result = self.result_parser[i].get_best_results()
            best_results = pd.concat(
                [best_results, pd.DataFrame({'name': [name] * len(best_result), 'best_result': best_result})])

        self._plot_all_with_data(
            data=best_results,
            target='best_result',
            title=title,
            x_label=x_label,
            y_label=y_label,
            color=color,
            kde_only=kde_only,
            figsize=figsize,
            plt_show=plt_show,
            save_path=save_path,
            ax=ax,
            **kwargs
        )

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
                         ax: Optional[plt.Axes] = None,
                         **kwargs
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
                               ax=ax,
                               **kwargs)

    def plot_result_time(self,
                         name: str,
                         title: str = 'Best Result Achieved',
                         x_label: str = 'Elapsed Time',
                         y_label: str = 'Energy Distance',
                         y_lim: Optional[tuple] = None,
                         color: str = 'C0',
                         figsize: tuple = (10, 5),
                         plt_show: bool = False,
                         save_path: Optional[str] = None,
                         ax: Optional[plt.Axes] = None,
                         **kwargs):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        color = self._get_color_palette([color])[0]
        result_time = pd.DataFrame(columns=['name', y_label, 'elapsed_time'])

        rv = self.result_parser[self.names.index(name)]

        for i in range(len(rv.min_result_matrix)):
            elapsed_time = np.nancumsum(rv.elapsed_time_matrix[i])
            result_time = pd.concat(
                [result_time, pd.DataFrame({'name': [name] * len(elapsed_time),
                                            y_label: rv.min_result_matrix[i],
                                            'elapsed_time': elapsed_time})])

        self._plot_result_time(
            data=result_time,
            x='elapsed_time',
            y=y_label,
            y_lim=y_lim,
            title=title,
            x_label=x_label,
            y_label=y_label,
            color=color,
            figsize=figsize,
            plt_show=plt_show,
            save_path=save_path,
            ax=ax,
            **kwargs
        )

        return ax

    @staticmethod
    def _plot_hist(data: np.ndarray,
                   title: str,
                   x_label: str,
                   y_label: str,
                   color: str,
                   name: str,
                   figsize: tuple,
                   plt_show: bool,
                   save_path: Optional[str],
                   ax: Optional[plt.Axes],
                   **kwargs
                   ) -> plt.Axes:
        """
        Plot the histogram of the data.
        :param data: The data to plot.
        :param title: The title of the plot.
        :param x_label: The x-axis label.
        :param y_label: The y-axis label.
        :param color: The color of the plot.
        :param figsize: The size of the figure.
        :param plt_show: Whether to show the plot.
        :param save_path: The path to save the plot. If None, the plot will not be saved.
        :param ax: The axes to plot on. If None, a new figure will be created.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        sns.kdeplot(
            data,
            ax=ax,
            color=color,
            label=name,
            **kwargs
        )

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        if plt_show:
            plt.show()

        if save_path is not None:
            plt.savefig(save_path)

        return ax

    @staticmethod
    def _plot_non_kde_hist(data: pd.DataFrame,
                           x: str,
                           title: str,
                           x_label: str,
                           y_label: str,
                           figsize: tuple,
                           plt_show: bool,
                           save_path: Optional[str],
                           ax: Optional[plt.Axes],
                           **kwargs
                           ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        sns.histplot(data=data,
                     hue='name',
                     x=x,
                     element='step',
                     # multiple=kwargs.get('multiple', 'stack'),
                     **kwargs
                     )

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        if plt_show:
            plt.show()

        if save_path is not None:
            plt.savefig(save_path)

        return ax

    @staticmethod
    def _plot_result_time(data: pd.DataFrame,
                          x: str,
                          y: str,
                          title: str,
                          x_label: str,
                          y_label: str,
                          y_lim: Optional[tuple] = None,
                          color: Optional[str] = None,
                          figsize: tuple = (10, 5),
                          plt_show: bool = True,
                          save_path: Optional[str] = None,
                          ax: Optional[plt.Axes] = None,
                          **kwargs):
        sns.histplot(
            data,
            x=x,
            y=y,
            ax=ax
        )
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        if y_lim is not None:
            ax.set_ylim(y_lim)

        if plt_show:
            plt.show()

        if save_path is not None:
            plt.savefig(save_path)

        return ax


class ResultVisualizerWithEarlyStopping(ResultVisualizer):
    def __init__(self,
                 result_paths: list[str],
                 metric: Optional[str] = None,
                 mode: str = 'min',
                 names: Optional[list[str]] = None,
                 cutoff: Optional[int] = None,
                 k: int = 10,
                 warmup: int = 50):
        """
        This class is used to visualize results from multiple runs with different configurations.
        It includes early stopping criteria.
        :param result_paths: A list of paths to the directories containing the results.
        :param metric: The metric to be visualized. If None, the results should be a float value. If not None,
        the results should be a dictionary containing the metric as the key.
        :param names: The names of the different configurations.
        :param mode: The mode to be used for the metric. Can be 'min' or 'max'.
        :param cutoff: The number of trials to be used for the metric. If None, all trials will be used.
        :param k: After k trials, if the results do not improve, the early stopping criteria is met.
        :param warmup: The number of trials to warm up before applying the early stopping criteria.
        """
        super().__init__(result_paths, metric, mode, names, cutoff)
        self.k = k
        self.warmup = warmup

    def plot_early_stopping_idx(self,
                                title: str = 'Early Stopping Index',
                                x_label: str = 'Number of Trials',
                                y_label: str = 'Frequency',
                                color: Optional[list] = None,
                                kde_only: bool = True,
                                figsize: tuple = (10, 5),
                                plt_show: bool = False,
                                save_path: Optional[str] = None,
                                ax: Optional[plt.Axes] = None,
                                **kwargs
                                ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        color = self._get_color_palette(color)

        early_stopping_idxs = pd.DataFrame(columns=['name', 'early_stopping_idx'])

        for i, name in enumerate(self.names):
            early_stopping_idx = self.result_parser[i].early_stopping_criteria(k=self.k, warmup=self.warmup)
            early_stopping_idxs = pd.concat(
                [early_stopping_idxs, pd.DataFrame({'name': [name] * len(early_stopping_idx),
                                                    'early_stopping_idx': early_stopping_idx})])

        self._plot_all_with_data(
            data=early_stopping_idxs,
            target='early_stopping_idx',
            title=title,
            x_label=x_label,
            y_label=y_label,
            color=color,
            kde_only=kde_only,
            figsize=figsize,
            plt_show=plt_show,
            save_path=save_path,
            ax=ax,
            **kwargs
        )

        return ax

    def plot_early_stopping_result(self,
                                   title: str = 'Early Stopping Result',
                                   x_label: str = 'Energy Distance',
                                   y_label: str = 'Frequency',
                                   color: Optional[list] = None,
                                   kde_only: bool = True,
                                   figsize: tuple = (10, 5),
                                   plt_show: bool = False,
                                   save_path: Optional[str] = None,
                                   ax: Optional[plt.Axes] = None,
                                   **kwargs
                                   ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        color = self._get_color_palette(color)

        early_stopping_results = pd.DataFrame(columns=['name', 'early_stopping_result'])

        for i, name in enumerate(self.names):
            early_stopping_idx = self.result_parser[i].early_stopping_criteria(k=self.k, warmup=self.warmup)
            early_stopping_result = self.result_parser[i].min_result_matrix[
                np.arange(len(early_stopping_idx)), early_stopping_idx - 1]
            early_stopping_results = pd.concat(
                [early_stopping_results, pd.DataFrame({'name': [name] * len(early_stopping_result),
                                                       'early_stopping_result': early_stopping_result})])

        self._plot_all_with_data(
            data=early_stopping_results,
            target='early_stopping_result',
            title=title,
            x_label=x_label,
            y_label=y_label,
            color=color,
            kde_only=kde_only,
            figsize=figsize,
            plt_show=plt_show,
            save_path=save_path,
            ax=ax,
            **kwargs
        )

        return ax

    def plot_early_stopping_elapsed_time(self,
                                         title: str = 'Early Stopping Elapsed Time',
                                         x_label: str = 'Elapsed Time',
                                         y_label: str = 'Frequency',
                                         color: Optional[list] = None,
                                         kde_only: bool = True,
                                         figsize: tuple = (10, 5),
                                         plt_show: bool = False,
                                         save_path: Optional[str] = None,
                                         ax: Optional[plt.Axes] = None,
                                         **kwargs
                                         ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        color = self._get_color_palette(color)
        early_stopping_elapsed_times = pd.DataFrame(columns=['name', 'early_stopping_elapsed_time'])

        for i, name in enumerate(self.names):
            early_stopping_idx = self.result_parser[i].early_stopping_criteria(k=self.k, warmup=self.warmup)
            early_stopping_elapsed_time = np.zeros(len(early_stopping_idx))
            for j, idx in enumerate(early_stopping_idx):
                early_stopping_elapsed_time[j] = np.nansum(self.result_parser[i].elapsed_time_matrix[j][:idx])
            early_stopping_elapsed_times = pd.concat(
                [early_stopping_elapsed_times, pd.DataFrame({'name': [name] * len(early_stopping_elapsed_time),
                                                             'early_stopping_elapsed_time': early_stopping_elapsed_time})])

        self._plot_all_with_data(
            data=early_stopping_elapsed_times,
            target='early_stopping_elapsed_time',
            title=title,
            x_label=x_label,
            y_label=y_label,
            color=color,
            kde_only=kde_only,
            figsize=figsize,
            plt_show=plt_show,
            save_path=save_path,
            ax=ax,
            **kwargs
        )

        return ax

    def plot_early_stopping_metric_score(self,
                                         title: str = 'Results',
                                         x_label: str = 'Number of Trials',
                                         y_label: str = 'Metric',
                                         color: Optional[list[str]] = None,
                                         y_lim: Optional[tuple] = None,
                                         ci: bool = True,
                                         ci_multiplier: float = 1.96,
                                         alpha: float = 0.2,
                                         figsize: tuple = (10, 5),
                                         plt_show: bool = True,
                                         save_path: Optional[str] = None,
                                         ax: Optional[plt.Axes] = None
                                         ) -> plt.Axes:
        color = self._get_color_palette(color)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        for i, result_parser in enumerate(self.result_parser):
            mean, sd = result_parser.get_early_stopping_mean_sd(k=self.k, warmup=self.warmup)

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
