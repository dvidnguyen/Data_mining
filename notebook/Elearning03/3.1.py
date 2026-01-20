# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Import các hàm tạo dữ liệu giả lập
from sklearn.datasets import make_circles, make_classification, make_moons

# Import các thuật toán phân loại (Classifiers) khác nhau
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

# Danh sách tên các thuật toán sẽ hiển thị trên biểu đồ
names = [
    "Nearest Neighbors",   # Thuật toán tìm láng giềng gần nhất
    "Linear SVM",          # Support Vector Machine (nhân tuyến tính)
    "RBF SVM",             # Support Vector Machine (nhân RBF - đường cong)
    "Gaussian Process",    # Quy trình Gaussian
    "Decision Tree",       # Cây quyết định
    "Random Forest",       # Rừng ngẫu nhiên (nhiều cây quyết định)
    "Neural Net",          # Mạng nơ-ron nhân tạo
    "AdaBoost",            # Thuật toán tăng cường (Boosting)
    "Naive Bayes",         # Xác suất Bayes ngây thơ
    "QDA",                 # Phân tích phân biệt bậc hai
]

# Khởi tạo các đối tượng thuật toán tương ứng với danh sách tên ở trên

classifiers = [
    KNeighborsClassifier(3), # K=3 láng giềng
    SVC(kernel="linear", C=0.025, random_state=42),
    SVC(gamma=2, C=1, random_state=42),
    GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    DecisionTreeClassifier(max_depth=5, random_state=42), # Cây sâu tối đa 5 tầng
    RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1, random_state=42
    ),
    MLPClassifier(alpha=1, max_iter=1000, random_state=42), # Mạng nơ-ron chạy tối đa 1000 vòng lặp
    AdaBoostClassifier(random_state=42),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

# --- TẠO DỮ LIỆU (DATASETS) ---

# 1. Tạo dữ liệu phân loại ngẫu nhiên (Linearly Separable - Có thể chia cắt bằng đường thẳng)
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape) # Thêm nhiễu vào dữ liệu để bài toán thực tế hơn
linearly_separable = (X, y)

# Danh sách 3 bộ dữ liệu để thử nghiệm:
datasets = [
    make_moons(noise=0.3, random_state=0),       # Dữ liệu hình 2 mặt trăng khuyết
    make_circles(noise=0.2, factor=0.5, random_state=1), # Dữ liệu hình tròn lồng nhau
    linearly_separable,                          # Dữ liệu phân tách tuyến tính
]

# Tạo khung hình lớn (Figure) để vẽ
figure = plt.figure(figsize=(27, 9))
i = 1 # Biến đếm vị trí của biểu đồ con (subplot)

# --- VÒNG LẶP CHÍNH ---
# Vòng lặp 1: Duyệt qua từng bộ dữ liệu (Dataset)
for ds_cnt, ds in enumerate(datasets):
    # Lấy dữ liệu X (tọa độ điểm) và y (nhãn màu đỏ/xanh)
    X, y = ds
    
    # Chia dữ liệu: 60% để học (Train), 40% để thi (Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    # Tìm giới hạn khung hình (min, max) để vẽ biểu đồ cho đẹp
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Cấu hình màu sắc: Đỏ và Xanh dương
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])

    # --- VẼ CỘT ĐẦU TIÊN: DỮ LIỆU GỐC ---
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data") # Đặt tiêu đề cho cột đầu tiên
    
    # Vẽ các điểm dùng để Train
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    # Vẽ các điểm dùng để Test (mờ hơn một chút - alpha=0.6)
    ax.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
    )
    # Ẩn các trục tọa độ x, y cho đỡ rối
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # Vòng lặp 2: Duyệt qua từng thuật toán (Classifier) để chạy thử trên bộ dữ liệu hiện tại
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

        # Tạo một quy trình (Pipeline):
        # Bước 1: StandardScaler -> Chuẩn hóa dữ liệu về cùng một tỷ lệ (rất quan trọng)
        # Bước 2: clf -> Chạy thuật toán phân loại
        clf = make_pipeline(StandardScaler(), clf)
        
        # HUẤN LUYỆN MÔ HÌNH (Học từ dữ liệu Train)
        clf.fit(X_train, y_train)
        
        # CHẤM ĐIỂM MÔ HÌNH (Thi trên dữ liệu Test)
        score = clf.score(X_test, y_test)
        
        # Vẽ ranh giới quyết định (Decision Boundary) - Vùng màu nền xanh/đỏ
        DecisionBoundaryDisplay.from_estimator(
            clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
        )

        # Vẽ lại các điểm Train lên trên
        ax.scatter(
            X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
        )
        # Vẽ lại các điểm Test lên trên
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_test,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.6,
        )

        # Căn chỉnh khung hình
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        
        # Nếu là dòng đầu tiên thì hiện tên thuật toán
        if ds_cnt == 0:
            ax.set_title(name)
            
        # In điểm số (độ chính xác) ở góc dưới bên phải biểu đồ
        ax.text(
            x_max - 0.3,
            y_min + 0.3,
            ("%.2f" % score).lstrip("0"), # Định dạng số, ví dụ .95 thay vì 0.95
            size=15,
            horizontalalignment="right",
        )
        i += 1

# Tự động căn chỉnh khoảng cách giữa các biểu đồ
plt.tight_layout()
# Hiển thị cửa sổ kết quả
plt.show()