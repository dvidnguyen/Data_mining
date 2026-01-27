1. Chọn ngẫu nhiên K điểm làm các tâm cụm ban đầu (Centroids).

2. Lặp lại cho đến khi không còn sự thay đổi về tâm cụm (hoặc đạt số bước lặp tối đa):

   a. Bước gán (Assignment):
      Với mỗi điểm dữ liệu x_i trong X:
      - Tính khoảng cách từ x_i đến K tâm cụm.
      - Gán x_i vào cụm có tâm gần nó nhất.

   b. Bước cập nhật (Update):
      Với mỗi cụm k từ 1 đến K:
      - Tính giá trị trung bình (mean) của tất cả các điểm trong cụm đó.
      - Cập nhật điểm trung bình này làm tâm cụm mới.

3. Trả về kết quả các cụm.