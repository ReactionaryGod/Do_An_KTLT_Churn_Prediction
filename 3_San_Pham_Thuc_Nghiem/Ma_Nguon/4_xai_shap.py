import pandas as pd
import os
import joblib
import shap
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(current_dir, '..', 'Du_Lieu', 'Da_Xu_Ly')
img_dir = os.path.join(current_dir, '..', '..', '2_Bao_Cao_Hoc_Thuat', 'Hinh_Anh_Bao_Cao')

print("Đang tải mô hình và dữ liệu...")
model = joblib.load(os.path.join(processed_dir, 'xgboost_model.pkl'))
scaler = joblib.load(os.path.join(processed_dir, 'scaler.pkl'))
X_test_raw = pd.read_csv(os.path.join(processed_dir, 'X_test_raw.csv'))

X_test_scaled = scaler.transform(X_test_raw)

print("Đang tính toán giá trị SHAP (Có thể mất vài giây)...")
explainer = shap.Explainer(model)
shap_values = explainer(X_test_scaled)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_raw, show=False) # Dùng X_test_raw để hiện đúng tên cột và giá trị thực tế
plt.title("Phân tích tầm quan trọng của các đặc trưng (SHAP Values)", fontsize=14, pad=20)
plt.tight_layout()

shap_img_path = os.path.join(img_dir, 'XAI_SHAP_Summary.png')
plt.savefig(shap_img_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Đã lưu biểu đồ SHAP tại: {shap_img_path}")