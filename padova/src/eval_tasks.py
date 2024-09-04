import argparse
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from eval_distance import infer_embeddings



def eval_classification(model_path: Path):
    print(">> evaluating tasks for model: ", model_path)

    X_train, y_train = infer_embeddings(model_path, train_ds=True)
    X_test, y_test = infer_embeddings(model_path, train_ds=False)

    print("shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    results = dict()

    print("training knn...")
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    knn_score = knn.score(X_test, y_test)
    print('KNN:', knn_score)
    results['knn'] = knn_score

    print("training svm...")
    svm = SVC()
    svm.fit(X_train, y_train)
    svm_score = svm.score(X_test, y_test)
    print('SVM:', svm_score)
    results['svm'] = svm_score

    print("training linear svm...")
    lsvm = LinearSVC()
    lsvm.fit(X_train, y_train)
    lsvm_score = lsvm.score(X_test, y_test)
    print('Linear SVM:', lsvm_score)
    results['lsvm'] = lsvm_score

    print("training random forest...")
    RF = RandomForestClassifier()
    RF.fit(X_train, y_train)
    rf_score = RF.score(X_test, y_test)
    print('RF:', rf_score)
    results['rf'] = rf_score

    with open(model_path.parent / "eval_class_results.csv", 'w') as f:
        for classifier, score in results.items():
            f.write(f"{classifier}, {score * 100: .3f}")



def generate_tsne(model_path: Path):
    print(">> generating tsne for model: ", model_path)

    X_test, y_test = infer_embeddings(model_path, train_ds=False)
    print("shapes:", X_test.shape, y_test.shape)

    # subset = y_test < 100 # first 10 classes
    # X_test = X_test[subset]
    # y_test = y_test[subset]

    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    X_tsne = tsne.fit_transform(X_test)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_test.astype(int), cmap='tab10', s=3)
    plt.legend(*scatter.legend_elements(), title="person (up to 10)")
    plt.title('t-sne')
    plt.savefig(model_path.parent / f"tsne.png", dpi=200)
    plt.close()



if __name__ == '__main__':
    ps = argparse.ArgumentParser()
    ps.add_argument('--path', type=str, required=True, help='path to the model')
    ps.add_argument('-r', '--recursive', type=bool, default=False, action=argparse.BooleanOptionalAction, help="evaluate all model in path (must be a folder)")
    ps.add_argument('-w', type=str, choices=['class', 'tsne'], default='class')
    args = ps.parse_args()

    model_paths = None

    if not args.recursive:
        model_path = Path(args.path)
        assert(model_path.is_file())
        model_paths = [model_path]
        print("selected:", model_path)
    else:
        folder_path = Path(args.path)
        assert(folder_path.is_dir())
        model_paths = list(folder_path.rglob("*.pth"))

        print("selected recursive:")
        for mp in model_paths:
            print(mp)


    for mp in model_paths:
        if args.w == 'class':
            eval_classification(mp)
        elif args.w == 'tsne':
            generate_tsne(mp)
        else:
            raise ValueError("invalid -w")

   