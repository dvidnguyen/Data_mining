# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
# BÀI TẬP 3.3 Nhận dạng chữ viết tay - chọn classifier tốt nhất

# ---
#
# **Họ và tên:** Nguyễn Quang Linh
# **MSSV:** 038205003016

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Import thêm load_digits và PCA (để giảm chiều dữ liệu)
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

# Giữ nguyên các import cũ của bạn
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, random_state=42),
    SVC(gamma=2, C=1, random_state=42),
    GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    DecisionTreeClassifier(max_depth=5, random_state=42),
    RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1, random_state=42
    ),
    MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    AdaBoostClassifier(random_state=42),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

# --- PHẦN SỬA ĐỔI DUY NHẤT: DATASET VỚI PCA ---

# 1. Load dữ liệu Digits
digits = load_digits()
X, y = digits.data, digits.target

# 2. Dùng PCA nén từ 64 chiều xuống 2 chiều (để vẽ được hình)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 3. Đưa vào list datasets (chỉ chạy 1 bộ này thôi)
datasets = [
    (X_pca, y)
]

# Thiết lập lại khung hình cho gọn vì chỉ có 1 dòng
figure = plt.figure(figsize=(27, 3))
i = 1

# --- VÒNG LẶP (GIỮ NGUYÊN CODE CŨ) ---
for ds_cnt, ds in enumerate(datasets):
    X, y = ds

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Lưu ý: Digits có 10 màu (lớp), nhưng cm_bright cũ chỉ có 2 màu.
    # Code vẫn chạy nhưng màu sắc các điểm sẽ bị lặp lại.
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])

    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data (PCA)")

    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.nipy_spectral, edgecolors="k")
    ax.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.nipy_spectral, alpha=0.6, edgecolors="k"
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # response_method="predict" giúp vẽ được đa lớp tốt hơn
        DecisionBoundaryDisplay.from_estimator(
            clf, X, cmap=plt.cm.nipy_spectral, alpha=0.8, ax=ax, eps=0.5, response_method="predict"
        )

        ax.scatter(
            X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.nipy_spectral, edgecolors="k"
        )
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_test,
            cmap=plt.cm.nipy_spectral,
            edgecolors="k",
            alpha=0.6,
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())

        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(
            x_max - 0.3,
            y_min + 0.3,
            ("%.2f" % score).lstrip("0"),
            size=15,
            horizontalalignment="right",
        )
        i += 1

plt.tight_layout()
plt.show()