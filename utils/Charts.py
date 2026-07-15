# Support functions for TS plotting
# Author: Jacob Cybulski, ironfrown[at]gmail.com
# Aims: Provide support for quantum time series analysis
# Date: 2024

import os
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import set_loglevel
from IPython.display import clear_output
from scipy.interpolate import make_interp_spline

from IPython.display import clear_output

### Bit to list translation for data entry and state interpretation
#   Note: PennyLane interprets qubits state in reverse order than Qiskit
#         These functions are a copy of functions in Circuits.py

### Transform int number to a list of bits, bit 0 comes first
def bin_int_to_list(a, n_bits):
    a = int(a)
    a_list = [int(i) for i in f'{a:0{n_bits}b}']
    # a_list.reverse()
    return np.array(a_list)

### Transform a list of bits to an int number, bit 0 comes first
def bin_list_to_int(bin_list):
    b = list(bin_list)
    # b.reverse()
    return int("".join(map(str, b)), base=2)


##### PL histograms

### Plot probability distribution
#   probs: list or tensor
#   thres: all probs less that threshold will not be plotted
def plot_hist(probs, scale=None, figsize=(8, 6), dpi=72, th=-10000, xlim=None, ylim=None, bottom=0, 
              labels=None, xlabel='Results', ylabel='Probability', title='Measurement Outcomes'):

    # Prepare data
    n_probs = len(probs)
    n_digits = len(bin_int_to_list(n_probs, 1)) # 1 means as many digits as required
    if labels is None: labels = [f'{n:0{n_digits}b}' for n in np.arange(n_probs)]

    # Filter out the prob values below threshold
    pairs = [(p, l) for (p, l) in zip(probs, labels) if p >= th]
    probs = [p for (p, l) in pairs]
    labels = [l for (p, l) in pairs]

    # Plot the results
    fig, ax=plt.subplots(figsize=figsize, dpi=dpi)
    ax.bar(labels, probs)
    plt.axhline(y=0, color="lightgray", linestyle='-')
    ax.set_title(title)
    if xlim is not None: plt.xlim(xlim)
    if ylim is not None: plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=60)
    if scale is not None:
        dpi = fig.get_dpi()
        fig.set_dpi(dpi*scale)
    plt.show()


### Plot two probability distributions side-by-side
#   probs_1, probs_2: Two tensors of probabilities
#   thres: all probs less that threshold will not be plotted
def plot_compare_hist(probs_1, probs_2, scale=None, figsize=(8, 6), dpi=72, th=0, 
                      title_1='Measurement Outcomes 1', title_2='Measurement Outcomes 2',
                      xlabel_1='Results', xlabel_2='Results',
                      ylabel_1='Probability', ylabel_2='Probability'):

    # Prepare data
    n_probs_1 = len(probs_1)
    n_digits_1 = len(bin_int_to_list(n_probs_1, 1)) # 1 means as many digits as required
    labels_1 = [f'{n:0{n_digits_1}b}' for n in np.arange(n_probs_1)]
    n_probs_2 = len(probs_2)
    n_digits_2 = len(bin_int_to_list(n_probs_2, 1)) # 1 means as many digits as required
    labels_2 = [f'{n:0{n_digits_2}b}' for n in np.arange(n_probs_2)]

    # Filter out the prob values below threshold
    pairs_1 = [(p, l) for (p, l) in zip(probs_1, labels_1) if p >= th]
    probs_1 = [p for (p, l) in pairs_1]
    labels_1 = [l for (p, l) in pairs_1]
    pairs_2 = [(p, l) for (p, l) in zip(probs_2, labels_2) if p >= th]
    probs_2 = [p for (p, l) in pairs_2]
    labels_2 = [l for (p, l) in pairs_2]

    # Plot the results
    fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    axs[0].bar(labels_1, probs_1)
    axs[0].set_title(title_1)
    axs[0].set_xlabel(xlabel_1)
    axs[0].set_ylabel(ylabel_1)
    axs[0].tick_params(labelrotation=60)
    axs[1].bar(labels_2, probs_2)
    axs[1].set_title(title_2)
    axs[1].set_xlabel(xlabel_2)
    axs[1].set_ylabel(ylabel_2)
    axs[1].tick_params(labelrotation=60)

    if scale is not None:
        dpi = fig.get_dpi()
        fig.set_dpi(dpi*scale)
    plt.show()

### Exponential Moving Target used to smooth the lines 
def smooth_movtarg(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value        
    return smoothed

### Plot performance measures
def meas_plot(meas_vals, rcParams=(8, 4), yscale='linear', log_interv=1, task='min',
                  backplot=False, back_color='linen', smooth_weight=0.9, save_plot=None,
                  meas='cost', title_pref='', xlim=None, ylim=None):
        
    if task == 'min':
        opt_cost = min(meas_vals)
        x_of_opt = np.argmin(meas_vals)
    else:
        opt_cost = max(meas_vals)
        x_of_opt = np.argmax(meas_vals)
    iter = len(meas_vals)
    smooth_fn = smooth_movtarg(meas_vals, smooth_weight)
    clear_output(wait=True)
    plt.rcParams["figure.figsize"] = rcParams
    plt.title(f'{title_pref} {meas} vs iteration '+('with smoothing ' if smooth_weight>0 else ' '))
    plt.xlabel(f'Iteration (best {meas}={np.round(opt_cost, 4)} @ iter# {x_of_opt*log_interv})')
    plt.ylabel(f'{meas.title()}')
    if xlim is not None: plt.xlim(xlim)
    if ylim is not None: plt.ylim(ylim)
    plt.axvline(x=x_of_opt*log_interv, color="lightgray", linestyle='--')
    plt.yscale(yscale)
    if backplot:
        plt.plot([x*log_interv for x in range(len(meas_vals))], meas_vals, color=back_color) # lightgray
    plt.plot([x*log_interv for x in range(len(smooth_fn))], smooth_fn, color='black')
    if save_plot is not None:
        os.makedirs(os.path.dirname(save_plot), exist_ok=True)
        ext = os.path.splitext(save_plot)[1][1:]
        plt.savefig(save_plot, format=ext)
    plt.show()


### Plot a list of series, where each series may start at a differ X point
#   y_list: a list of series with equidistant points
#   X_list: a list of starting X points for each series
#   labels, color, lines, marks: Plot features for each series
#   other: standard plot properties
def multi_plot_flat_ts_with_start(
    y_list, X_list=None, labels=None, colors=None, lines=None, markers=None, marker_colors=None,
    xlim=None, ylim=None, rcParams=(12, 6), xlabel='Range', ylabel='Target value',
    legend_cols=3, title='Time series plot', save_plot=None):

    # labels=['Target function', 'Training data', 'Test data', 'Fitted model', 'Model predictions'],
    # colors=['lightblue', 'lightblue', 'pink', 'blue', 'red'],
    # linestyles=['dashed', 'solid', 'solid', 'dashed', 'dashed'],

    if len(y_list) == 0:
        print(f'*** Error: the list of data to plot cannot be empty')
        return
        
    # Missing main parameters
    if X_list is None or len(X_list) == 0:
        X_list = [0] # [[i for i in ]]
    if len(X_list) < len(y_list):
        ts_start = X_list[-1]+len(y_list[len(X_list)])
        for ts in y_list[len(X_list):]:
            X_list.append(ts_start)
            ts_start += len(ts)

    if labels is None or len(labels) == 0:
        labels = [f'Plot {0:02d}']
    if len(labels) < len(y_list):
        for i in range(len(labels), len(y_list)):
            labels.append(f'Plot {i:02d}')

    cmap = matplotlib.colormaps['Set1']
    map_colors = cmap.colors+cmap.colors+cmap.colors
    if colors is None or len(colors) == 0:
        colors = [map_colors[0]]
    if len(colors) < len(y_list):
        for i in range(len(colors), len(y_list)):
            colors.append(map_colors[i])

    if marker_colors is None or len(marker_colors) == 0:
        marker_colors = colors
    if len(marker_colors) < len(y_list):
        for i in range(len(marker_colors), len(y_list)):
            marker_colors.append(colors[i])

    if lines is None or len(lines) == 0:
        lines = ['solid']
    if len(lines) < len(y_list):
        lines = lines+['solid']*(len(y_list)-len(lines))
    
    if markers is None or len(markers) == 0:
        markers = ['none']
    if len(markers) < len(y_list):
        markers = markers+['none']*(len(y_list)-len(markers))
    
    # Parameter values
    if rcParams is not None:
        plt.rcParams["figure.figsize"] = rcParams        
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Calculate the X_range for each list
    
    for i in range(len(y_list)):
        xfirst = X_list[i]
        xlast = X_list[i]+len(y_list[i])
        xrange = range(xfirst, xlast)
        plt.plot(xrange, y_list[i], linestyle=lines[i], marker=markers[i], 
                 color=colors[i], mec=colors[i], mfc=marker_colors[i], label=labels[i])
        if i > 0:
            plt.axvline(x = X_list[i]-0.5, color = 'lightgray', linestyle='dashed')
    
    plt.legend(loc='best', ncol=legend_cols)
    
    if save_plot is not None:
        os.makedirs(os.path.dirname(save_plot), exist_ok=True)
        ext = os.path.splitext(save_plot)[1][1:]
        plt.savefig(save_plot, format=ext)
    plt.show()    


### Plot a list of series, where each series may start at a differ X point
#   X_list: a list of X coordinates for the series points
#   y_list: a list of series of series point values
#   vert_lines: a list of vertical line styles, default none
#   labels, color, lines, marks: Plot features for each series
#   other: standard plot properties
def multi_plot_flat_ts(
    X_list, y_list, vert_lines=[], vert_line_color='lightgray',
    labels=None, colors=None, lines=None, markers=None, marker_colors=None,
    xlim=None, ylim=None, rcParams=(12, 6), dpi=72, xlabel='Range', ylabel='Target value',
    legend_cols=3, title='Time series plot', save_plot=None):

    # labels=['Target function', 'Training data', 'Test data', 'Fitted model', 'Model predictions'],
    # colors=['lightblue', 'lightblue', 'pink', 'blue', 'red'],
    # linestyles=['dashed', 'solid', 'solid', 'dashed', 'dashed'],

    # Small distance from the last value
    sigma = 0.05
    
    # Incompatible X and y lists
    if len(X_list) != len(y_list):
        print(f'*** Error: the list of data to plot cannot be empty')
        return

    if labels is None or len(labels) == 0:
        labels = [f'Plot {0:02d}']
    if len(labels) < len(y_list):
        for i in range(len(labels), len(y_list)):
            labels.append(f'Plot {i:02d}')

    cmap = matplotlib.colormaps['Set1']
    map_colors = cmap.colors+cmap.colors+cmap.colors
    if colors is None or len(colors) == 0:
        colors = [map_colors[0]]
    if len(colors) < len(y_list):
        for i in range(len(colors), len(y_list)):
            colors.append(map_colors[i])

    if marker_colors is None or len(marker_colors) == 0:
        marker_colors = colors
    if len(marker_colors) < len(y_list):
        for i in range(len(marker_colors), len(y_list)):
            marker_colors.append(colors[i])

    if lines is None or len(lines) == 0:
        lines = ['solid']
    if len(lines) < len(y_list):
        lines = lines+['solid']*(len(y_list)-len(lines))
    
    if markers is None or len(markers) == 0:
        markers = ['none']
    if len(markers) < len(y_list):
        markers = markers+['none']*(len(y_list)-len(markers))
    
    # Parameter values
    plt.rcParams["figure.figsize"] = rcParams   
    plt.rcParams["figure.dpi"] = dpi

    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Calculate the X_range for each list
    
    for i in range(len(y_list)):
        if len(vert_lines) >= i+1:
            plt.axvline(x = X_list[i][0]-sigma, color = vert_line_color, linestyle=vert_lines[i])
        plt.plot(X_list[i], y_list[i], linestyle=lines[i], marker=markers[i], 
                 color=colors[i], mec=colors[i], mfc=marker_colors[i], label=labels[i])
    
    plt.legend(loc='best', ncol=legend_cols)
    
    if save_plot is not None:
        os.makedirs(os.path.dirname(save_plot), exist_ok=True)
        ext = os.path.splitext(save_plot)[1][1:]
        plt.savefig(save_plot, format=ext)
    plt.show()    

### Plot performance measures
def smooth_plot(meas_vals, rcParams=(8, 4), yscale='linear', log_interv=1, task='min',
                  backplot=False, back_color='linen', smooth_weight=0.9, save_plot=None,
                  meas='cost', title_pref='', xlim=None, ylim=None):
        
    if task == 'min':
        opt_cost = min(meas_vals)
        x_of_opt = np.argmin(meas_vals)
    else:
        opt_cost = max(meas_vals)
        x_of_opt = np.argmax(meas_vals)
    iter = len(meas_vals)
    smooth_fn = smooth_movtarg(meas_vals, smooth_weight)
    clear_output(wait=True)
    plt.rcParams["figure.figsize"] = rcParams
    plt.title(f'{title_pref} {meas} vs iteration '+('with smoothing ' if smooth_weight>0 else ' '))
    plt.xlabel(f'Iteration (best {meas}={np.round(opt_cost, 4)} @ iter# {x_of_opt*log_interv})')
    plt.ylabel(f'{meas.title()}')
    if xlim is not None: plt.xlim(xlim)
    if ylim is not None: plt.ylim(ylim)
    plt.axvline(x=x_of_opt*log_interv, color="lightgray", linestyle='--')
    plt.yscale(yscale)
    if backplot:
        plt.plot([x*log_interv for x in range(len(meas_vals))], meas_vals, color=back_color) # lightgray
    plt.plot([x*log_interv for x in range(len(smooth_fn))], smooth_fn, color='black')
    if save_plot is not None:
        os.makedirs(os.path.dirname(save_plot), exist_ok=True)
        ext = os.path.splitext(save_plot)[1][1:]
        plt.savefig(save_plot, format=ext)
    plt.show()


### Exponential Moving Target used to smooth the lines 
def smooth_movtarg(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value        
    return smoothed

def spline_multi_plot(xvals, train_vals_list, test_vals_list, rcParams=(8, 4), dpi=100, yscale='linear', 
    log_interv=1, task='min', backplot=False, back_color='linen', k=2, 
    save_plot=None, meas='cost', title=None, xlim=None, ylim=None,
    train_cols=None, test_cols=None):

    if title is None:
        title = f'QRC Impact of Input Leakage on MSE'
    if train_cols is None:
        # Specify colours for background and mean (solid) / median (dashed)
        # Where background is the shadow from min to max across group
        train_cols = ['skyblue', 'royalblue']
    if test_cols is None:
        # Specify colours for background and mean (solid) / median (dashed)
        # Where background is the shadow from min to max across group
        test_cols = ['pink', 'crimson']

    # Collect smooth curves
    train_smooth_list = []
    test_smooth_list = []
    X = np.array(xvals)
    X_smooth = np.linspace(X.min(), X.max(), 300)
    for group_no in range(len(train_vals_list)):
        y_train = np.array(train_vals_list[group_no])
        y_test = np.array(test_vals_list[group_no])
        train_spline = make_interp_spline(X, y_train, k=k)  # Cubic spline (k=3)
        y_train_smooth = train_spline(X_smooth)
        test_spline = make_interp_spline(X, y_test, k=k)  # Cubic spline (k=3)
        y_test_smooth = test_spline(X_smooth)
        train_smooth_list.append(y_train_smooth)
        test_smooth_list.append(y_test_smooth)

    # Plot the chart
    clear_output(wait=True)
    plt.rcParams["figure.figsize"] = rcParams
    plt.rcParams["figure.dpi"] = dpi
    plt.title(title)
    plt.xlabel(f'Leakage (qubits#={n_qubits}, layers#={n_layers}, scalers: weights={weight_scaler}, biases={bias_scaler}, phase={phase_scaler})')
    plt.ylabel(f'MSE')
    if xlim is not None: 
        plt.xlim(xlim)
    else: plt.xlim(0, 1)
    if ylim is not None: plt.ylim(ylim)
    
    # Plot training data
    train_mean_MSE = np.array(train_mse_scores_list).min(axis=1).mean()
    select_label = f'Train: mean MSE={train_mean_MSE:0.6f}'
    select = np.array(train_smooth_list)
    plt.plot(X_smooth, select.mean(axis=0), color = train_cols[1], label=select_label)
    plt.fill_between(X_smooth, select.max(axis=0), select.min(axis=0), 
                     color = train_cols[0], alpha = 0.2)

    # Plot test data
    test_mean_MSE = np.array(test_mse_scores_list).min(axis=1).mean()
    select_label = f'Test: mean MSE={test_mean_MSE:0.6f}'
    select = np.array(test_smooth_list)
    plt.plot(X_smooth, select.mean(axis=0), color = test_cols[1], label=select_label)
    plt.fill_between(X_smooth, select.max(axis=0), select.min(axis=0), 
                     color = test_cols[0], alpha = 0.2)

    start, end = plt.xlim()
    plt.xticks(np.arange(start, end, 0.1))
    plt.legend(ncol=1)

    if save_plot is not None:
        os.makedirs(os.path.dirname(save_plot), exist_ok=True)
        ext = os.path.splitext(save_plot)[1][1:]
        plt.savefig(save_plot, format=ext)
    plt.show()