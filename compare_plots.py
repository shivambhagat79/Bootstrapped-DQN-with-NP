import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
import string

def plot_comparison(df1, df2, x_col, y_col, y_std_col=None, xlabel=None, ylabel=None, title=None, legend_labels=['File 1', 'File 2'], output_dir='comparison', color=['b', 'r'], rolling_window=5):
    """
    Plots a comparison line graph for a given metric from two dataframes.

    Args:
        df1 (pd.DataFrame): DataFrame for the first file.
        df2 (pd.DataFrame): DataFrame for the second file.
        x_col (str): Column name for the x-axis.
        y_col (str): Column name for the y-axis.
        y_std_col (str, optional): Column name for y-axis standard deviation for error bands. Defaults to None.
        xlabel (str, optional): Label for the x-axis. Defaults to x_col.
        ylabel (str, optional): Label for the y-axis. Defaults to y_col.
        title (str, optional): Title for the plot. Defaults to 'Comparison of {y_col}'.
        legend_labels (list, optional): Labels for the legend. Defaults to ['File 1', 'File 2'].
        output_dir (str, optional): Directory to save the plot. Defaults to 'comparison'.
        color (list, optional): Colors for the lines. Defaults to ['b', 'r'].
        rolling_window (int, optional): Window size for rolling average. Set to 0 or 1 to disable. Defaults to 5.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure() # Create a new figure for each plot

    xlabel = xlabel if xlabel else x_col.replace('_', ' ').title()
    ylabel = ylabel if ylabel else y_col.replace('_', ' ').title()
    title = title if title else f'Comparison of {ylabel}'
    filename = title.replace(' ', '_') + ".png"
    filepath = os.path.join(output_dir, filename)

    dfs = [df1, df2]
    has_data = [False, False]

    for i, df in enumerate(dfs):
        if x_col in df.columns and y_col in df.columns:
            # Drop rows where y_col is NaN or empty, as these can cause issues with plotting
            # Ensure x_col is sorted for rolling average to make sense if needed, though usually step counts are monotonic
            df_cleaned = df[[x_col, y_col]].sort_values(by=x_col).dropna(subset=[y_col])
            if not df_cleaned.empty:
                x_data = df_cleaned[x_col]
                y_data = df_cleaned[y_col]

                # Ensure data is numeric
                x_data = pd.to_numeric(x_data, errors='coerce')
                y_data = pd.to_numeric(y_data, errors='coerce')

                # Drop NaNs introduced by coercion
                valid_indices = ~np.isnan(x_data) & ~np.isnan(y_data)
                x_data = x_data[valid_indices]
                y_data = y_data[valid_indices]

                # Apply rolling average if window > 1
                if rolling_window > 1 and len(x_data) >= rolling_window:
                    # Use pandas rolling mean - it handles NaNs and alignment well
                    # Note: Rolling mean reduces the number of points. We plot the mean against the *end* of the window's x-value.
                    y_data_smooth = y_data.rolling(window=rolling_window, min_periods=1).mean()
                    # Keep x_data aligned with the smoothed y_data
                    plot_x = x_data
                    plot_y = y_data_smooth
                else:
                    plot_x = x_data
                    plot_y = y_data


                if not plot_x.empty:
                    plt.plot(plot_x, plot_y, linewidth=1.5, color=color[i], label=legend_labels[i])
                    has_data[i] = True

                    # Add error bands if std dev column is provided and exists
                    # Apply rolling average to std dev as well if smoothing is enabled
                    if y_std_col and y_std_col in df.columns:
                         # Align std dev data with cleaned y_data
                        y_std_data = pd.to_numeric(df.loc[df_cleaned.index, y_std_col], errors='coerce')
                        y_std_data = y_std_data[valid_indices] # Filter based on valid x and y
                        valid_std_indices = ~np.isnan(y_std_data)

                        if np.all(valid_std_indices):
                            # Ensure all arrays are aligned and have valid data for fill_between
                            aligned_x = plot_x[valid_std_indices]
                            aligned_y = plot_y[valid_std_indices]
                            aligned_std = y_std_data[valid_std_indices]

                            # Smooth std dev if needed (using rolling mean of variance might be more correct, but mean is simpler)
                            if rolling_window > 1 and len(aligned_std) >= rolling_window:
                                aligned_std_smooth = aligned_std.rolling(window=rolling_window, min_periods=1).mean()
                            else:
                                aligned_std_smooth = aligned_std

                            # Ensure lengths match after potential rolling average
                            min_len = min(len(aligned_x), len(aligned_y), len(aligned_std_smooth))
                            aligned_x = aligned_x[-min_len:]
                            aligned_y = aligned_y[-min_len:]
                            aligned_std_smooth = aligned_std_smooth[-min_len:]


                            plt.fill_between(aligned_x,
                                             aligned_y - 2 * aligned_std_smooth,
                                             aligned_y + 2 * aligned_std_smooth,
                                             color=color[i], alpha=0.1)
                        else:
                             print(f"Warning: Skipping error band for {legend_labels[i]} in '{title}' due to missing/invalid std dev data.")

            else:
                 print(f"Warning: No valid data for {y_col} in {legend_labels[i]} for plot '{title}'.")
        else:
            print(f"Warning: Column '{x_col}' or '{y_col}' not found in {legend_labels[i]} for plot '{title}'.")

    if any(has_data):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc="best")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filepath)
        print(f"Saved plot: {filepath}")
    else:
        print(f"Skipping plot '{title}' as no data was available from either file.")

    plt.close() # Close the figure to free memory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare performance data from two CSV files.")
    parser.add_argument("csv_file1", help="Path to the first CSV file.")
    parser.add_argument("csv_file2", help="Path to the second CSV file.")
    parser.add_argument("-l1", "--label1", default="File 1", help="Label for the first file in legends.")
    parser.add_argument("-l2", "--label2", default="File 2", help="Label for the second file in legends.")
    parser.add_argument("-o", "--output_dir", default="comparison", help="Directory to save comparison plots.")
    parser.add_argument("-rw", "--rolling_window", type=int, default=5, help="Window size for rolling average (0 or 1 to disable). Default: 5")

    args = parser.parse_args()

    # --- Load Data ---
    try:
        df1 = pd.read_csv(args.csv_file1)
        # Clean column names (remove leading/trailing spaces)
        df1.columns = df1.columns.str.strip()
    except FileNotFoundError:
        print(f"Error: File not found - {args.csv_file1}")
        exit(1)
    except Exception as e:
        print(f"Error reading {args.csv_file1}: {e}")
        exit(1)

    try:
        df2 = pd.read_csv(args.csv_file2)
        df2.columns = df2.columns.str.strip()
    except FileNotFoundError:
        print(f"Error: File not found - {args.csv_file2}")
        exit(1)
    except Exception as e:
        print(f"Error reading {args.csv_file2}: {e}")
        exit(1)

    legend_labels = [args.label1, args.label2]
    output_dir = args.output_dir
    rolling_window = args.rolling_window

    # --- Generate Plots ---
    # Determine a common game name if possible for titles (optional, based on file paths)
    try:
        game_name_part1 = os.path.basename(os.path.dirname(args.csv_file1)).split('_')[0]
        game_name_part2 = os.path.basename(os.path.dirname(args.csv_file2)).split('_')[0]
        if game_name_part1 == game_name_part2:
             plot_title_prefix = string.capwords(game_name_part1.replace("_", " ")) + " -"
        else:
             plot_title_prefix = ""
    except Exception:
        plot_title_prefix = ""


    # Evaluation Scores (with error bands)
    plot_comparison(df1, df2, x_col='eval_steps', y_col='eval_rewards', y_std_col='eval_stds',
                    title=f'{plot_title_prefix} Mean Evaluation Scores', xlabel='Evaluation Steps', ylabel='Score',
                    legend_labels=legend_labels, output_dir=output_dir, rolling_window=rolling_window)

    # Highest Evaluation Score
    plot_comparison(df1, df2, x_col='eval_steps', y_col='highest_eval_score',
                    title=f'{plot_title_prefix} Highest Evaluation Score', xlabel='Evaluation Steps', ylabel='Highest Score',
                    legend_labels=legend_labels, output_dir=output_dir, rolling_window=rolling_window)

    # Average Training Rewards
    plot_comparison(df1, df2, x_col='steps', y_col='avg_rewards',
                    title=f'{plot_title_prefix} Average Training Rewards', xlabel='Training Steps', ylabel='Average Reward',
                    legend_labels=legend_labels, output_dir=output_dir, rolling_window=rolling_window)

    # Q-Values
    plot_comparison(df1, df2, x_col='steps', y_col='q_record',
                    title=f'{plot_title_prefix} Max Q-Values', xlabel='Training Steps', ylabel='Q-Value',
                    legend_labels=legend_labels, output_dir=output_dir, rolling_window=rolling_window)

    # Episode Reward
    plot_comparison(df1, df2, x_col='steps', y_col='episode_reward',
                    title=f'{plot_title_prefix} Episode Reward', xlabel='Training Steps', ylabel='Episode Reward',
                    legend_labels=legend_labels, output_dir=output_dir, rolling_window=rolling_window)

    # Episode Steps
    plot_comparison(df1, df2, x_col='steps', y_col='episode_step',
                    title=f'{plot_title_prefix} Episode Steps', xlabel='Training Steps', ylabel='Steps per Episode',
                    legend_labels=legend_labels, output_dir=output_dir, rolling_window=rolling_window)

    # Epsilon
    plot_comparison(df1, df2, x_col='steps', y_col='eps_list',
                    title=f'{plot_title_prefix} Epsilon Decay', xlabel='Training Steps', ylabel='Epsilon',
                    legend_labels=legend_labels, output_dir=output_dir, rolling_window=rolling_window)

    # Episode Loss
    plot_comparison(df1, df2, x_col='steps', y_col='episode_loss',
                    title=f'{plot_title_prefix} Episode Loss', xlabel='Training Steps', ylabel='Loss',
                    legend_labels=legend_labels, output_dir=output_dir, rolling_window=rolling_window)

    # Episode Times
    plot_comparison(df1, df2, x_col='steps', y_col='episode_times',
                    title=f'{plot_title_prefix} Episode Duration', xlabel='Training Steps', ylabel='Time (s)',
                    legend_labels=legend_labels, output_dir=output_dir, rolling_window=rolling_window)

    # Relative Episode Times
    plot_comparison(df1, df2, x_col='steps', y_col='episode_relative_times',
                    title=f'{plot_title_prefix} Cumulative Episode Time', xlabel='Training Steps', ylabel='Cumulative Time (s)',
                    legend_labels=legend_labels, output_dir=output_dir, rolling_window=rolling_window)

    # --- Optional Plots (Check if columns exist) ---
    optional_cols = {
        'int_rew': 'Intrinsic Reward',
        'total_rew': 'Total Reward (Extrinsic + Intrinsic)',
        'auxillary_loss': 'Auxiliary Loss' # Note: Typo might be 'auxiliary_loss'
    }

    for col, name in optional_cols.items():
         # Check both possible spellings for auxiliary
        actual_col = col
        if col == 'auxillary_loss':
            if 'auxiliary_loss' in df1.columns or 'auxiliary_loss' in df2.columns:
                actual_col = 'auxiliary_loss' # Use correct spelling if found
            elif 'auxillary_loss' not in df1.columns and 'auxillary_loss' not in df2.columns:
                 continue # Skip if neither spelling exists

        if actual_col in df1.columns or actual_col in df2.columns:
            plot_comparison(df1, df2, x_col='steps', y_col=actual_col,
                            title=f'{plot_title_prefix} {name}', xlabel='Training Steps', ylabel=name,
                            legend_labels=legend_labels, output_dir=output_dir, rolling_window=rolling_window)
        else:
            print(f"Skipping optional plot for '{name}' (column '{actual_col}' not found).")

    print("\nComparison plotting complete.")
