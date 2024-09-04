import argparse
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier

from eval_distance import infer_embeddings


if __name__ == '__main__':
    ps = argparse.ArgumentParser()
    ps.add_argument('--path', type=str, default=None, help='path to the model')
    args = ps.parse_args()

    model_path = Path(args.path)

    print(">> evaluating tasks for model: ", model_path)

    x_train, y_train = infer_embeddings(model_path, train_ds=True)
    x_test, y_test = infer_embeddings(model_path, train_ds=False)

    print("shapes:", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    results = dict()

    print("training knn...")
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    knn_score = knn.score(x_test, y_test)
    print('KNN:', knn_score)
    results['knn'] = knn_score

    print("training svm...")
    svm = SVC()
    svm.fit(x_train, y_train)
    svm_score = svm.score(x_test, y_test)
    print('SVM:', svm_score)
    results['svm'] = svm_score

    print("training linear svm...")
    lsvm = LinearSVC()
    lsvm.fit(x_train, y_train)
    lsvm_score = lsvm.score(x_test, y_test)
    print('Linear SVM:', lsvm_score)
    results['lsvm'] = lsvm_score

    print("training random forest...")
    RF = RandomForestClassifier()
    RF.fit(x_train, y_train)
    rf_score = RF.score(x_test, y_test)
    print('RF:', rf_score)
    results['rf'] = rf_score

    with open(model_path.parent / "eval_class_results.csv", 'w') as f:
        for classifier, score in results.items():
            f.write(f"{classifier}, {score * 100: .3f}")