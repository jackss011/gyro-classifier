import argparse
from pathlib import Path
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import cluster
from sklearn import metrics


from eval_distance import infer_embeddings



def eval_classification(model_path: Path):
    print("\n>> evaluating classification tasks for model: ", model_path)

    resutls_path = model_path.parent / "eval_class_results.csv"
    if resutls_path.exists():
        df = pd.read_csv(resutls_path, header=None)
        df.columns = ['classifier', 'score']
    else:
        df = pd.DataFrame(columns=['classifier', 'score'])

    prev_classifiers = list(df['classifier'])
    print("already classified with:", prev_classifiers)

    X_train, y_train = infer_embeddings(model_path, train_ds=True)
    X_test, y_test = infer_embeddings(model_path, train_ds=False)

    print("shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    results = dict()

    if 'knn' not in prev_classifiers:
        print("training knn...")
        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)
        knn_score = knn.score(X_test, y_test)
        print('KNN:', knn_score)
        results['knn'] = knn_score * 100

    if 'svm' not in prev_classifiers:
        print("training svm...")
        svm = SVC()
        svm.fit(X_train, y_train)
        svm_score = svm.score(X_test, y_test)
        print('SVM:', svm_score)
        results['svm'] = svm_score * 100

    if 'lsvm' not in prev_classifiers:
        # print("training linear svm...")
        # lsvm = LinearSVC(max_iter=100)
        # lsvm.fit(X_train, y_train)
        # lsvm_score = lsvm.score(X_test, y_test)
        # print('Linear SVM:', lsvm_score)
        # results['lsvm'] = lsvm_score * 100
        pass

    if 'mlp' not in prev_classifiers:
        print('training multi-layer perceptron...')
        MLP = MLPClassifier()
        MLP.fit(X_train, y_train)
        mlp_score = MLP.score(X_test, y_test)
        print('MLP:', mlp_score)
        results['mlp'] = mlp_score * 100

    if 'rf' not in prev_classifiers:
        print("training random forest...")
        RF = RandomForestClassifier()
        RF.fit(X_train, y_train)
        rf_score = RF.score(X_test, y_test)
        print('RF:', rf_score)
        results['rf'] = rf_score * 100

    if len(results) > 0:
        df_new = pd.DataFrame(results.items(), columns=['classifier', 'score'])
        df = pd.concat((df, df_new))
        print(df)
    df.to_csv(resutls_path, header=None, index=False)



def generate_tsne(model_path: Path):
    print("\n>> generating tsne for model: ", model_path)

    X_test, y_test = infer_embeddings(model_path, train_ds=False)
    print("shapes:", X_test.shape, y_test.shape)

    # subset = y_test < 100 # first 10 classes
    # X_test = X_test[subset]
    # y_test = y_test[subset]

    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    X_tsne = tsne.fit_transform(X_test)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_test.astype(int), cmap='tab10', s=3)
    plt.legend(*scatter.legend_elements(), title="person")
    plt.title('t-sne')
    plt.savefig(model_path.parent / f"tsne.png", dpi=200)
    plt.close()


def eval_clustering(model_path: Path):
    print("\n>> evaluating clustering for model: ", model_path)

    X_test, y_test = infer_embeddings(model_path, train_ds=False)
    print("shapes:", X_test.shape, y_test.shape)
    num_classes = len(set(y_test))
    print("num classes:", num_classes)

    kmeans = cluster.KMeans(n_clusters=num_classes, random_state=42)
    y_pred = kmeans.fit_predict(X_test)

    # dbscan = cluster.DBSCAN()
    # y_pred = dbscan.fit_predict(X_test)
    results = dict()

    sh_score = metrics.silhouette_score(X_test, y_pred)
    print("silhouette score:", sh_score)
    results['silhouette'] = sh_score

    rand_score = metrics.rand_score(y_test, y_pred)
    print("rand score:", rand_score)
    results['rand'] = rand_score

    adj_rand_score = metrics.adjusted_rand_score(y_test, y_pred)
    print("adj rand score:", adj_rand_score)
    results['rand_adj'] = adj_rand_score

    df = pd.DataFrame(dict(value=results))
    df.to_csv(model_path.parent / 'eval_cluster_results.csv', index=True, header=False)

    return sh_score, rand_score, adj_rand_score


if __name__ == '__main__':
    ps = argparse.ArgumentParser()
    ps.add_argument('--path', type=str, required=True, help='path to the model')
    ps.add_argument('-t', '--task', type=str, choices=['class', 'tsne', 'cluster'], default='class')
    args = ps.parse_args()

    path = Path(args.path)
    model_paths = None

    if path.is_file():
        model_paths = [Path(path)]
        print("selected single model:", model_paths[0])
    elif path.is_dir():
        model_paths = list(path.rglob("*.pth"))

        print("selected model folder containing:")
        for mp in model_paths:
            print(mp)

    for mp in model_paths:
        print("\n\n")
        
        if args.task == 'class':
            eval_classification(mp)
        elif args.task == 'tsne':
            generate_tsne(mp)
        elif args.task == 'cluster':
            os.environ["OMP_NUM_THREADS"] = "15" # suggested by warning message on windows
            eval_clustering(mp)
        else:
            raise ValueError(f"invalid --task: {args.task}")

   