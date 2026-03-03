import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="E-Commerce Churn Dashboard", page_icon="📊", layout="wide")
st.title("📊 Hệ thống Dự báo Rời bỏ Khách hàng (Customer Churn)")
st.markdown("---")


@st.cache_resource
def load_assets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(current_dir, '..', 'Du_Lieu', 'Da_Xu_Ly')

    model = joblib.load(os.path.join(processed_dir, 'xgboost_model.pkl'))
    scaler = joblib.load(os.path.join(processed_dir, 'scaler.pkl'))

    df_new = pd.read_csv(os.path.join(processed_dir, 'X_test_raw.csv'))
    return model, scaler, df_new


try:
    model, scaler, df_new = load_assets()
except Exception as e:
    st.error(f"Lỗi tải dữ liệu. Hãy chắc chắn bạn đã chạy 3 file code trước đó! Chi tiết: {e}")
    st.stop()

X_scaled = scaler.transform(df_new)

churn_probs = model.predict_proba(X_scaled)[:, 1]

df_display = df_new.copy()
df_display['Risk_Score (%)'] = (churn_probs * 100).round(2)
df_display['Dự đoán'] = df_display['Risk_Score (%)'].apply(lambda x: 'Nguy cơ cao' if x > 50 else 'An toàn')

st.subheader("1. Tổng quan tình trạng Khách hàng")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Tổng số khách hàng quét", len(df_display))
with col2:
    high_risk_count = len(df_display[df_display['Risk_Score (%)'] > 50])
    st.metric("Số khách hàng Nguy cơ cao", high_risk_count, delta_color="inverse")
with col3:
    avg_risk = df_display['Risk_Score (%)'].mean()
    st.metric("Tỷ lệ rủi ro trung bình", f"{avg_risk:.1f}%")

st.markdown("---")

st.subheader("2. Danh sách Khách hàng & Điểm rủi ro")
st.write("Bộ phận CSKH sử dụng bảng này để ưu tiên gọi điện/tặng voucher cho khách có màu Đỏ/Cam.")

df_display_sorted = df_display.sort_values(by='Risk_Score (%)', ascending=False)


def color_risk(val):
    color = 'red' if val > 70 else 'orange' if val > 40 else 'green'
    return f'color: {color}; font-weight: bold'


st.dataframe(
    df_display_sorted.style.map(color_risk, subset=['Risk_Score (%)']),
    use_container_width=True,
    height=300
)
st.markdown("---")

st.subheader("3. Phân tích nguyên nhân rời bỏ (AI Explainability)")
st.write("Chọn một khách hàng cụ thể để xem tại sao hệ thống AI lại dự đoán người này sắp bỏ đi.")

selected_idx = st.selectbox("Chọn Mã Khách Hàng (Index):", df_display_sorted.index.tolist())

# Nút bấm để phân tích
if st.button("🔍 Phân tích nguyên nhân"):
    with st.spinner("Hệ thống đang trích xuất đồ thị SHAP..."):
        customer_data = X_scaled[selected_idx].reshape(1, -1)

        explainer = shap.Explainer(model)
        shap_values_single = explainer(customer_data)

        shap_values_single.feature_names = df_new.columns.tolist()

        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(shap_values_single[0], show=False)
        plt.tight_layout()

        st.pyplot(fig)

        st.info(
            "💡 **Hướng dẫn đọc biểu đồ:** Thanh màu Đỏ đẩy điểm rủi ro tăng lên (Lý do khách rời bỏ). Thanh màu Xanh kéo điểm rủi ro xuống (Lý do khách muốn ở lại).")