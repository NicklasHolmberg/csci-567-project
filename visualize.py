import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Directory containing your JSON files
test_folder = 'test'  # Replace with the actual path to your test folder

# Loop through each JSON file in the test folder
for filename in os.listdir(test_folder):
    if filename.endswith('.json'):
        file_path = os.path.join(test_folder, filename)

        # Load JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Visualize Confusion Matrix
        conf_matrix = np.array(data['test_confusion_matrix'])

        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix for {filename}')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()

        # Visualize Classification Report (Precision, Recall, F1-Score)
        classification_report = data['test_classification_report']

        # Extract class labels and values
        labels = [label for label in classification_report if isinstance(classification_report[label], dict)]
        precision = [classification_report[label]['precision'] for label in labels]
        recall = [classification_report[label]['recall'] for label in labels]
        f1_score = [classification_report[label]['f1-score'] for label in labels]

        # Plot Precision, Recall, F1-Score
        x = np.arange(len(labels))  # the label locations
        width = 0.2  # the width of the bars

        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width, precision, width, label='Precision')
        rects2 = ax.bar(x, recall, width, label='Recall')
        rects3 = ax.bar(x + width, f1_score, width, label='F1-Score')

        # Add some text for labels, title, and custom x-axis tick labels
        ax.set_xlabel('Classes')
        ax.set_title(f'Precision, Recall, F1-Score by Class for {filename}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.legend()

        # Show plot
        plt.tight_layout()
        plt.show()