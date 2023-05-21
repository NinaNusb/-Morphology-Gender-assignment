import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def exp_1_visual():
    """ experiment 1 visualization """

    languages = ['Polish', 'German', 'French', 'Spanish']
    
    knn_iterative_df = pd.read_csv('../data/knn_per_language_results.csv')
    per_iterative_df = pd.read_csv('../data/perceptron_per_language_results.csv')


    x_axis = np.arange(1, 6)  # the number of encoded letters (first 5 columns)

    # Iterate over each language
    for i, language in enumerate(languages):
        knn_scores = knn_iterative_df[language].values[:5]  # Extract KNN scores for the language (first 5 columns)
        perceptron_scores = per_iterative_df[language].values[:5]  # Extract Perceptron scores for the language (first 5 columns)

        # Create a new figure and subplot for each language
        fig, ax = plt.subplots()

        ax.plot(x_axis, knn_scores, color='#003f5c', label=f'KNN')
        ax.plot(x_axis, perceptron_scores, color='#ffa600', linestyle='--', label=f'Perceptron')

        # Add labels, legend, and title for each language plot
        ax.set_xlabel('Number of Encoded Letters')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.set_title(f'Figure {i+1}: Performance of KNN and Perceptron Models - {language}')

        # Set x-axis ticks to integers
        ax.set_xticks(x_axis)

        # Display the plot for each language
        plt.savefig(f'../visuals/experiment_1_{language}_graph.png')


def exp_3_visual(results):
    """ experiment 3 visualization """

    categories = ['Baseline', 'KNN', 'Perceptron']
    performance_values = [results[cat] for cat in categories]

    x = np.arange(len(categories))
    width = 0.5

    fig, ax = plt.subplots()

    # the colors for each bar
    colors = ['LightGray', '#003f5c', '#ffa600']

    # Plot the bars with the specified colors
    bars = ax.bar(x, performance_values, width, color=colors)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Accuracy')
    ax.set_title('Figure 5: Comparing Model Performance with Baseline Performance')

    # Display the percentage value on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2%}', ha='center', va='bottom')

    plt.savefig('../visuals/experiment_3_graph.png')