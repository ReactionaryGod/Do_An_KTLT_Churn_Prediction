import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

current_dir = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(current_dir, '..', 'Du_Lieu', 'Da_Xu_Ly')
img_dir = os.path.join(current_dir, '..', '..', '2_Bao_Cao_Hoc_Thuat', 'Hinh_Anh_Bao_Cao')
os.makedirs(img_dir, exist_ok=True)

data_path = os.path.join(processed_dir, 'Churn_Modelling_Cleaned.csv')
df = pd.read_csv(data_path)


print("Đang vẽ biểu đồ EDA...")
sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Gender', hue='Exited', palette='Set2')
plt.title('Tỷ lệ rời bỏ khách hàng theo Giới tính')
plt.savefig(os.path.join(img_dir, 'EDA_Churn_by_Gender.png'), dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='Age', hue='Exited', multiple="stack", kde=True, palette='viridis')
plt.title('Phân phối Độ tuổi theo trạng thái Rời bỏ')
plt.savefig(os.path.join(img_dir, 'EDA_Age_Distribution.png'), dpi=300)
plt.close()

print("Đang mã hóa dữ liệu...")
df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})

df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

encoded_path = os.path.join(processed_dir, 'Churn_Encoded.csv')
df.to_csv(encoded_path, index=False)
print(f"Đã lưu dữ liệu mã hóa tại: {encoded_path}")
print(f"Đã lưu hình ảnh biểu đồ tại: {img_dir}")