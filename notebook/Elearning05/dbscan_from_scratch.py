"""
BÀI 5.2 - CÀI ĐẶT DBSCAN TỪ ĐẦU VÀ SO SÁNH VỚI SKLEARN
Môn: Khai Thác Dữ Liệu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs,make_circles
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# ============================================================================
# PHẦN 1: CÀI ĐẶT DBSCAN TỪ ĐẦU (KHÔNG DÙNG THƯ VIỆN)
# ============================================================================

class DBSCAN:
    """
    Thuật toán DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    Cài đặt từ đầu không sử dụng sklearn.
    """
    
    def __init__(self, eps=0.5, min_pts=5):
        """
        Khởi tạo tham số cho DBSCAN.
        
        Tham số:
            eps: Bán kính vùng lân cận (epsilon)
            min_pts: Số điểm tối thiểu để tạo thành vùng đặc (core point)
        """
        self.eps = eps
        self.min_pts = min_pts
        self.labels_ = None          # Nhãn cụm của mỗi điểm
        self.core_points_ = None     # Danh sách các điểm lõi
        self.border_points_ = None   # Danh sách các điểm biên
        self.noise_points_ = None    # Danh sách các điểm nhiễu
    
    def _tinh_khoang_cach(self, X):
        """
        Tính ma trận khoảng cách Euclidean giữa tất cả các cặp điểm.
        
        Tham số:
            X: Ma trận dữ liệu (n_samples, n_features)
        
        Trả về:
            Ma trận khoảng cách (n_samples, n_samples)
        """
        # Số lượng điểm dữ liệu
        n = len(X)
        
        # Khởi tạo ma trận khoảng cách
        khoang_cach = np.zeros((n, n))
        
        # Tính khoảng cách Euclidean cho từng cặp điểm
        for i in range(n):
            for j in range(i + 1, n):
                # Công thức Euclidean: sqrt(sum((x1 - x2)^2))
                d = np.sqrt(np.sum((X[i] - X[j]) ** 2))
                khoang_cach[i, j] = d
                khoang_cach[j, i] = d  # Ma trận đối xứng
        
        return khoang_cach
    
    def _tim_lan_can(self, ma_tran_kc, chi_so_diem):
        """
        Tìm tất cả các điểm nằm trong vùng lân cận eps của một điểm.
        
        Tham số:
            ma_tran_kc: Ma trận khoảng cách
            chi_so_diem: Chỉ số của điểm cần tìm lân cận
        
        Trả về:
            Danh sách chỉ số các điểm lân cận
        """
        # Lấy hàng khoảng cách từ điểm hiện tại đến tất cả điểm khác
        khoang_cach_tu_diem = ma_tran_kc[chi_so_diem]
        
        # Tìm các điểm có khoảng cách <= eps
        lan_can = np.where(khoang_cach_tu_diem <= self.eps)[0]
        
        return lan_can
    
    def _phan_loai_diem(self, X, ma_tran_kc):
        """
        PHASE 1: Phân loại các điểm thành CORE, BORDER, NOISE.
        
        Tham số:
            X: Ma trận dữ liệu
            ma_tran_kc: Ma trận khoảng cách
        
        Trả về:
            neighbors: Danh sách lân cận của mỗi điểm
            is_core: Mảng boolean đánh dấu điểm lõi
            is_border: Mảng boolean đánh dấu điểm biên
            is_noise: Mảng boolean đánh dấu điểm nhiễu
        """
        n = len(X)
        
        # Tìm lân cận cho tất cả các điểm
        neighbors = [self._tim_lan_can(ma_tran_kc, i) for i in range(n)]
        
        # Xác định điểm lõi (CORE): có >= min_pts điểm trong vùng eps
        is_core = np.array([len(neighbors[i]) >= self.min_pts for i in range(n)])
        
        # Xác định điểm biên (BORDER): không phải core nhưng nằm trong vùng eps của core
        is_border = np.zeros(n, dtype=bool)
        for i in range(n):
            if not is_core[i]:
                # Kiểm tra xem điểm i có là lân cận của điểm core nào không
                for j in neighbors[i]:
                    if is_core[j]:
                        is_border[i] = True
                        break
        
        # Xác định điểm nhiễu (NOISE): không phải core và không phải border
        is_noise = ~is_core & ~is_border
        
        return neighbors, is_core, is_border, is_noise
    
    def _mo_rong_cum(self, diem_bat_dau, neighbors, is_core, labels, cluster_id):
        """
        Mở rộng cụm từ một điểm lõi bằng thuật toán BFS.
        Tìm tất cả các điểm density-connected với điểm bắt đầu.
        
        Tham số:
            diem_bat_dau: Chỉ số điểm lõi bắt đầu
            neighbors: Danh sách lân cận của mỗi điểm
            is_core: Mảng đánh dấu điểm lõi
            labels: Mảng nhãn cụm
            cluster_id: ID của cụm hiện tại
        """
        # Sử dụng hàng đợi để duyệt BFS
        hang_doi = [diem_bat_dau]
        
        while hang_doi:
            # Lấy điểm đầu hàng đợi
            diem_hien_tai = hang_doi.pop(0)
            
            # Bỏ qua nếu đã được gán nhãn
            if labels[diem_hien_tai] != -1:
                continue
            
            # Gán nhãn cụm cho điểm hiện tại
            labels[diem_hien_tai] = cluster_id
            
            # Nếu là điểm lõi, thêm các lân cận chưa được gán nhãn vào hàng đợi
            if is_core[diem_hien_tai]:
                for diem_lan_can in neighbors[diem_hien_tai]:
                    if labels[diem_lan_can] == -1:
                        hang_doi.append(diem_lan_can)
    
    def fit(self, X):
        """
        Huấn luyện mô hình DBSCAN trên dữ liệu X.
        
        Tham số:
            X: Ma trận dữ liệu (n_samples, n_features)
        
        Trả về:
            self: Đối tượng DBSCAN đã được huấn luyện
        """
        n = len(X)
        
        # Bước 1: Tính ma trận khoảng cách
        print("Bước 1: Tính ma trận khoảng cách...")
        ma_tran_kc = self._tinh_khoang_cach(X)
        
        # Bước 2: Phân loại điểm thành CORE, BORDER, NOISE
        print("Bước 2: Phân loại điểm (Core, Border, Noise)...")
        neighbors, is_core, is_border, is_noise = self._phan_loai_diem(X, ma_tran_kc)
        
        # Lưu lại thông tin phân loại
        self.core_points_ = np.where(is_core)[0]
        self.border_points_ = np.where(is_border)[0]
        self.noise_points_ = np.where(is_noise)[0]
        
        # Bước 3: Gom cụm các điểm lõi (density-connected)
        print("Bước 3: Gom cụm các điểm lõi...")
        labels = np.full(n, -1)  # -1 = chưa gán nhãn / noise
        cluster_id = -1
        
        for i in range(n):
            # Chỉ xét các điểm lõi chưa được gán nhãn
            if is_core[i] and labels[i] == -1:
                cluster_id += 1
                # Mở rộng cụm từ điểm lõi này
                self._mo_rong_cum(i, neighbors, is_core, labels, cluster_id)
        
        # Bước 4: Gán điểm biên vào cụm của điểm lõi gần nhất
        print("Bước 4: Gán điểm biên vào cụm gần nhất...")
        for i in range(n):
            if is_border[i]:
                # Tìm điểm lõi gần nhất trong vùng lân cận
                cac_core_lan_can = [j for j in neighbors[i] if is_core[j]]
                if cac_core_lan_can:
                    # Chọn điểm lõi có khoảng cách nhỏ nhất
                    diem_core_gan_nhat = min(cac_core_lan_can, 
                                              key=lambda j: ma_tran_kc[i, j])
                    labels[i] = labels[diem_core_gan_nhat]
        
        self.labels_ = labels
        
        print(f"Hoàn thành! Tìm thấy {cluster_id + 1} cụm.")
        return self
    
    def fit_predict(self, X):
        """
        Huấn luyện mô hình và trả về nhãn cụm.
        
        Tham số:
            X: Ma trận dữ liệu
        
        Trả về:
            labels: Nhãn cụm của mỗi điểm (-1 = noise)
        """
        self.fit(X)
        return self.labels_


# ============================================================================
# PHẦN 2: SO SÁNH VỚI SKLEARN
# ============================================================================

def so_sanh_ket_qua(X, labels_custom, labels_sklearn):
    """
    So sánh kết quả giữa DBSCAN tự cài đặt và sklearn.
    
    Tham số:
        X: Dữ liệu gốc
        labels_custom: Nhãn từ DBSCAN tự cài đặt
        labels_sklearn: Nhãn từ sklearn DBSCAN
    """
    print("\n" + "=" * 60)
    print("SO SÁNH KẾT QUẢ")
    print("=" * 60)
    
    # Số cụm tìm được (không tính noise)
    n_clusters_custom = len(set(labels_custom)) - (1 if -1 in labels_custom else 0)
    n_clusters_sklearn = len(set(labels_sklearn)) - (1 if -1 in labels_sklearn else 0)
    
    # Số điểm nhiễu
    n_noise_custom = np.sum(labels_custom == -1)
    n_noise_sklearn = np.sum(labels_sklearn == -1)
    
    print(f"\n{'Chỉ số':<30} {'Tự cài đặt':<15} {'Sklearn':<15}")
    print("-" * 60)
    print(f"{'Số cụm tìm được:':<30} {n_clusters_custom:<15} {n_clusters_sklearn:<15}")
    print(f"{'Số điểm nhiễu:':<30} {n_noise_custom:<15} {n_noise_sklearn:<15}")
    
    # Tính các chỉ số đánh giá
    # ARI (Adjusted Rand Index): đo độ tương đồng giữa 2 phân cụm
    ari = adjusted_rand_score(labels_sklearn, labels_custom)
    
    # NMI (Normalized Mutual Information): đo lượng thông tin chung
    nmi = normalized_mutual_info_score(labels_sklearn, labels_custom)
    
    print(f"\n{'Độ tương đồng (so với sklearn):'}")
    print(f"  - Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"  - Normalized Mutual Info (NMI): {nmi:.4f}")
    
    if ari > 0.95:
        print("\n✓ Kết quả gần như giống hệt sklearn!")
    elif ari > 0.8:
        print("\n✓ Kết quả tương tự sklearn.")
    else:
        print("\n! Có sự khác biệt với sklearn.")


def ve_bieu_do(X, labels_custom, labels_sklearn, dbscan_custom):
    """
    Vẽ biểu đồ so sánh kết quả phân cụm.
    
    Tham số:
        X: Dữ liệu gốc
        labels_custom: Nhãn từ DBSCAN tự cài đặt
        labels_sklearn: Nhãn từ sklearn
        dbscan_custom: Đối tượng DBSCAN tự cài đặt
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Biểu đồ 1: Kết quả DBSCAN tự cài đặt
    ax1 = axes[0]
    scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=labels_custom, cmap='viridis', 
                           s=50, edgecolors='black', linewidth=0.5)
    ax1.set_title('DBSCAN Tự Cài Đặt', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    
    # Biểu đồ 2: Kết quả sklearn
    ax2 = axes[1]
    scatter2 = ax2.scatter(X[:, 0], X[:, 1], c=labels_sklearn, cmap='viridis',
                           s=50, edgecolors='black', linewidth=0.5)
    ax2.set_title('DBSCAN Sklearn', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    
    # Biểu đồ 3: Phân loại điểm (Core, Border, Noise)
    ax3 = axes[2]
    colors = np.zeros(len(X))
    colors[dbscan_custom.core_points_] = 1    # Core = 1
    colors[dbscan_custom.border_points_] = 2  # Border = 2
    colors[dbscan_custom.noise_points_] = 0   # Noise = 0
    
    scatter3 = ax3.scatter(X[:, 0], X[:, 1], c=colors, cmap='coolwarm',
                           s=50, edgecolors='black', linewidth=0.5)
    ax3.set_title('Phân Loại Điểm', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Feature 1')
    ax3.set_ylabel('Feature 2')
    
    # Thêm chú thích
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
               markersize=10, label=f'Noise ({len(dbscan_custom.noise_points_)})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='black', markersize=10, 
               label=f'Core ({len(dbscan_custom.core_points_)})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=10, label=f'Border ({len(dbscan_custom.border_points_)})')
    ]
    ax3.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    plt.savefig('dbscan_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nĐã lưu biểu đồ vào file 'dbscan_comparison.png'")


# ============================================================================
# PHẦN 3: CHƯƠNG TRÌNH CHÍNH
# ============================================================================

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Bước 1: Tạo dữ liệu mẫu
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("DBSCAN - SO SÁNH TỰ CÀI ĐẶT VỚI SKLEARN")
    print("=" * 60)
    
    # Tạo dataset hình trăng lưỡi liềm (phù hợp với DBSCAN)
    print("\n1. Tạo dữ liệu make_moons...")
    X, y_true = make_moons(n_samples=1000, noise=0.2, random_state=36)
    print(f"   - Số điểm dữ liệu: {len(X)}")
    print(f"   - Số chiều: {X.shape[1]}")
    
    # -------------------------------------------------------------------------
    # Bước 2: Thiết lập tham số
    # -------------------------------------------------------------------------
    # eps: Bán kính vùng lân cận
    # min_pts: Số điểm tối thiểu để tạo thành core point
    eps = 0.15
    min_pts = 5
    print(f"\n2. Tham số DBSCAN:")
    print(f"   - eps (bán kính): {eps}")
    print(f"   - min_pts (điểm tối thiểu): {min_pts}")
    
    # -------------------------------------------------------------------------
    # Bước 3: Chạy DBSCAN tự cài đặt
    # -------------------------------------------------------------------------
    print("\n3. Chạy DBSCAN tự cài đặt...")
    print("-" * 40)
    
    dbscan_custom = DBSCAN(eps=eps, min_pts=min_pts)
    labels_custom = dbscan_custom.fit_predict(X)
    
    print(f"\nKết quả:")
    print(f"   - Điểm Core: {len(dbscan_custom.core_points_)}")
    print(f"   - Điểm Border: {len(dbscan_custom.border_points_)}")
    print(f"   - Điểm Noise: {len(dbscan_custom.noise_points_)}")
    
    # -------------------------------------------------------------------------
    # Bước 4: Chạy DBSCAN sklearn để so sánh
    # -------------------------------------------------------------------------
    print("\n4. Chạy DBSCAN sklearn...")
    
    dbscan_sklearn = SklearnDBSCAN(eps=eps, min_samples=min_pts)
    labels_sklearn = dbscan_sklearn.fit_predict(X)
    
    # -------------------------------------------------------------------------
    # Bước 5: So sánh kết quả
    # -------------------------------------------------------------------------
    so_sanh_ket_qua(X, labels_custom, labels_sklearn)
    
    # -------------------------------------------------------------------------
    # Bước 6: Vẽ biểu đồ
    # -------------------------------------------------------------------------
    print("\n5. Vẽ biểu đồ so sánh...")
    ve_bieu_do(X, labels_custom, labels_sklearn, dbscan_custom)
    
    # -------------------------------------------------------------------------
    # Thử với dataset khác: make_blobs
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("THỬ VỚI DATASET MAKE_BLOBS")
    print("=" * 60)
    
    # Tạo dữ liệu blobs
    X2, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.5, random_state=42)
    
    # Tham số mới phù hợp với blobs
    eps2 = 0.8
    min_pts2 = 5
    
    print(f"\nTham số: eps={eps2}, min_pts={min_pts2}")
    
    # Chạy cả 2 phiên bản
    dbscan_custom2 = DBSCAN(eps=eps2, min_pts=min_pts2)
    labels_custom2 = dbscan_custom2.fit_predict(X2)
    
    dbscan_sklearn2 = SklearnDBSCAN(eps=eps2, min_samples=min_pts2)
    labels_sklearn2 = dbscan_sklearn2.fit_predict(X2)
    
    # So sánh
    so_sanh_ket_qua(X2, labels_custom2, labels_sklearn2)
    
    print("\n" + "=" * 60)
    print("HOÀN THÀNH!")
    print("=" * 60)
