import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score
from imblearn.over_sampling import SMOTE

current_dir = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(current_dir, '..', 'Du_Lieu', 'Da_Xu_Ly')
model_dir = os.path.join(current_dir, '..', 'Da_Xu_Ly') # Lưu model ở đây cho tiện

df = pd.read_csv(os.path.join(processed_dir, 'Churn_Encoded.csv'))

X = df.drop('Exited', axis=1)
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Đang áp dụng SMOTE để cân bằng dữ liệu...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print("Đang huấn luyện mô hình XGBoost...")
model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

y_pred = model.predict(X_test_scaled)
print("\n=== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ===")
print(f"Độ chính xác (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print(f"Độ nhạy (Recall): {recall_score(y_test, y_pred):.4f}")
print("\nBáo cáo chi tiết:")
print(classification_report(y_test, y_pred))

joblib.dump(model, os.path.join(processed_dir, 'xgboost_model.pkl'))
joblib.dump(scaler, os.path.join(processed_dir, 'scaler.pkl'))
X_test.to_csv(os.path.join(processed_dir, 'X_test_raw.csv'), index=False) # Lưu X_test nguyên bản để vẽ SHAP cho đẹp
print(f"\nĐã lưu Model và Scaler thành công!")