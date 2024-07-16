from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
import NaiveBayes as nb
import NLP as data
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt



def check_data_dir(data_dir: str) -> None:
    dirs_to_check = [
        f'{data_dir}/training/ham',
        f'{data_dir}/training/spam',
        f'{data_dir}/test/ham',
        f'{data_dir}/test/spam',
    ]
    for path_str in dirs_to_check:
        path_obj = Path(path_str)
        if not path_obj.is_dir():
            raise NotADirectoryError(f"'{path_obj}' is not a valid directory")


def train(data_dir: str) -> nb.NaiveBayes:
    training_set_path = Path(f'{data_dir}/training')
    x, y = data.load_data(data_dir, subset='training')
    best_alpha = nb.optimize_hyperparameters(x, y)
    print(f"Best alpha: {best_alpha}")
    trained = nb.NaiveBayes(alpha = best_alpha) # the final model
    trained.fit(x,y)
    return trained


def test(final_model: nb.NaiveBayes, data_dir)-> nb.NaiveBayes:
    test_set_path = Path(f'{data_dir}/test')
    x,y = data.load_data(data_dir, subset='test')
    return nb.evaluate_model(x,y)
    


def main() -> None:
    parser = argparse.ArgumentParser(
        description="""
        Run naive Bayes spam classification on the specified dataset.
        """,
    )
    parser.add_argument(
        '-d', '--data-dir',
        metavar="DIR",
        default='./data/',
        help="use DIR as the dataset directory (default: %(default)s)",
    )
    args = parser.parse_args()
    data_dir = args.data_dir


    try:
        check_data_dir(data_dir)
    except NotADirectoryError as e:
        print(f"ERROR: {e}", file = sys.stderr)
        print("Please make sure your dataset directory is valid.",
            file = sys.stderr)
        print(f"Try '{sys.argv[0]} --help' for more information.")
        sys.exit(1)

    print("Training...")
    final_model = train(data_dir)
    
    print("Testing...")
    results = test(final_model, data_dir)
    
    
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Cross-Validation Score: {results['f1_score']:.4f}") # calculated using average of f1 scores
    print("Confusion Matrix:")
    print(results['confusion_matrix'])
    print("Additional Graphical Representation of Confusion Matrix has been provided")
    classnames = ['True Spam', 'True Ham']
    disp = ConfusionMatrixDisplay(confusion_matrix = results['confusion_matrix'], display_labels = classnames)
    disp.plot(cmap='Blues')
    plt.show()



if __name__ == "__main__":
    main()
