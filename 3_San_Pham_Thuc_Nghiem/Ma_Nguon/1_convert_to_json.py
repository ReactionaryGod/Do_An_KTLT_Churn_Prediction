import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

raw_dir = os.path.join(current_dir, '..', 'Du_Lieu', 'Tho')
processed_dir = os.path.join(current_dir, '..', 'Du_Lieu', 'Da_Xu_Ly')

os.makedirs(processed_dir, exist_ok=True)

data_path = os.path.join(raw_dir, 'Churn_Modelling.csv')
print(f"Đang đọc dữ liệu từ: {data_path}")
df = pd.read_csv(data_path)

df_cleaned = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

csv_path = os.path.join(processed_dir, 'Churn_Modelling_Cleaned.csv')
df_cleaned.to_csv(csv_path, index=False)
print(f"Đã lưu file CSV sạch tại: {csv_path}")

json_path = os.path.join(processed_dir, 'Churn_Modelling.json')
df_cleaned.to_json(json_path, orient='records', force_ascii=False, indent=4)
print(f"Đã lưu file JSON tại: {json_path}")