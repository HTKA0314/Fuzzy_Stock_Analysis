# 📈 FUZZY_STOCK_ANALYSIS: HỆ THỐNG DỰ ĐOÁN XU HƯỚNG GIÁ CỔ PHIẾU (HYBRID MODEL)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Dự án này triển khai một hệ thống lai (**Hybrid Model**) kết hợp **Fuzzy Logic (Tập mờ)** và **Random Forest (Học máy)** để dự đoán xu hướng giá cổ phiếu trên thị trường Việt Nam. Hệ thống sử dụng phương pháp luận kiểm thử nghiêm ngặt **Walk-Forward Validation (WFV)**.

---

## 🚀 1. Hướng dẫn Cài đặt và Khởi chạy

### 1.1. Cài đặt Thư viện

Chạy các lệnh sau trong Terminal hoặc cửa sổ dòng lệnh để cài đặt tất cả các thư viện cần thiết:

```bash
# Cài đặt các thư viện khoa học dữ liệu và Streamlit
!pip install scikit-fuzzy pandas matplotlib numpy streamlit seaborn ta

# Cài đặt thư viện vnstock phiên bản tương thích
!pip install vnstock==0.2.5
