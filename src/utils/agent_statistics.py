def common_plot_statrees(
    statrees_with_info: List[Tuple[Dict, dict]], output_dir: str, path: str = ""
):
    """
    Plot multiple statrees on the same graph for comparison.

    Args:
        statrees_with_info: List of tuples, each containing (statree, plot_info)
            where plot_info is a dict with keys:
            - 'color': color to use for this statree's lines
            - 'alpha': transparency level (0-1)
            - 'linewidth': width of the plotted line
            - 'linestyle': style of the line ('-', '--', ':', etc.)
            - 'label': label for the legend
            - 'is_mean': boolean indicating if this is a mean line (for styling)
            - Any other matplotlib Line2D property can be passed
        output_dir: Directory to save the plots
        path: Current path in the statree hierarchy for recursive calls
    """
    os.makedirs(output_dir, exist_ok=True)

    # First, find all unique keys across all statrees at this level
    all_keys = set()
    for statree, _ in statrees_with_info:
        all_keys.update(statree.keys())

    for key in all_keys:
        new_path = f"{path}/{key}" if path else key

        # Collect all dictionaries and lists at this key position
        dicts_at_key = []
        lists_at_key = []

        for statree, plot_info in statrees_with_info:
            if key in statree:
                value = statree[key]
                if isinstance(value, dict):
                    dicts_at_key.append((value, plot_info))
                elif isinstance(value, list):
                    lists_at_key.append((value, plot_info))

        # If we found dictionaries, recurse into them
        if dicts_at_key:
            common_plot_statrees(dicts_at_key, output_dir, new_path)

        # If we found lists, plot them
        if lists_at_key:
            plt.figure(figsize=(10, 6))

            # Plot each list with its styling
            for data_list, plot_info in lists_at_key:
                # Extract known parameters
                color = plot_info.get("color", "blue")
                alpha = plot_info.get("alpha", 1.0)
                linewidth = plot_info.get("linewidth", 1)
                linestyle = plot_info.get("linestyle", "-")
                label = plot_info.get("label", None)

                # Skip empty lists
                if not data_list:
                    continue

                # Clean up None values
                cleaned_data = [v if v is not None else 0 for v in data_list]

                # Create a plot options dictionary for matplotlib
                plot_options = {
                    "color": color,
                    "alpha": alpha,
                    "linewidth": linewidth,
                    "linestyle": linestyle,
                }

                # Add label only if provided
                if label:
                    plot_options["label"] = label

                # Add any other matplotlib parameters from plot_info
                for k, v in plot_info.items():
                    if k not in [
                        "color",
                        "alpha",
                        "linewidth",
                        "linestyle",
                        "label",
                        "is_mean",
                    ]:
                        plot_options[k] = v

                # Plot the data
                plt.plot(cleaned_data, **plot_options)

            # Add plot metadata
            plt.title(new_path)
            plt.xlabel("Iterations")
            plt.ylabel(key.replace("_", " ").title())

            # Add legend if we have labels
            if any(info.get("label") for _, info in lists_at_key):
                plt.legend()

            # Save the plot
            output_filename = os.path.join(
                output_dir, f"{new_path.replace('/', '_')}.png"
            )
            plt.savefig(output_filename)
            plt.close()


def plot_seed_averaged_stats(root_path, agent_names):
    """
    Plots seed-averaged statistics for given agents.

    This function is kept for backwards compatibility but uses the improved implementation.

    Args:
        root_path (str): Path to the directory containing seed data.
        agent_names (list): List of agent names to process.
    """
    seed_dirs = []
    # Identify all seed directories
    for date_dir in os.listdir(root_path):
        date_path = os.path.join(root_path, date_dir)
        if os.path.isdir(date_path):
            seed_dirs.extend(
                [
                    os.path.join(date_path, dir_name)
                    for dir_name in os.listdir(date_path)
                    if dir_name.startswith("seed")
                ]
            )

    for agent in agent_names:
        # Create output directory for averaged stats
        avg_stats_dir = os.path.join(root_path, "avg_seed_stats", agent)
        os.makedirs(avg_stats_dir, exist_ok=True)

        # Collect paths to all statistic files for this agent
        stat_paths = []
        for seed_dir in seed_dirs:
            stats_file = os.path.join(
                seed_dir, "statistics", agent, f"{agent}_stats.jsonl"
            )
            if os.path.exists(stats_file):
                stat_paths.append(stats_file)

        # Use the improved implementation
        if stat_paths:
            plot_seed_averaged_stats_improved(stat_paths, avg_stats_dir, agent)

    print(
        f"Seed-averaged plots saved successfully at {os.path.join(root_path, 'avg_seed_stats')}!"
    )
