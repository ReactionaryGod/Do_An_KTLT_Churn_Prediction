import os
import sys
import importlib.util
import pandas as pd
import joblib
import matplotlib.pyplot as plt


class ChurnModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.df_new = pd.DataFrame()
        self.expected_columns = []
        self.X_scaled = None
        self.churn_probs = None
        self.model_ready = False  # Backward-compatible alias for predictor_ready
        self.model_error = ""     # Backward-compatible alias for predictor_error
        self.predictor_ready = False
        self.predictor_error = ""
        self.shap_ready = False
        self.shap_error = ""
        self.load_assets()

    def get_data_path(self, filename):
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
            return os.path.join(base_path, 'Da_Xu_Ly', filename)
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            return os.path.abspath(os.path.join(current_dir, '..', '..', 'Du_Lieu', 'Da_Xu_Ly', filename))

    def load_assets(self):
        self.df_new = pd.read_csv(self.get_data_path('X_test_raw.csv'))
        self.expected_columns = list(self.df_new.columns)
        self.scaler = joblib.load(self.get_data_path('scaler.pkl'))
        self._update_shap_readiness()

        try:
            self.model = joblib.load(self.get_data_path('xgboost_model.pkl'))
            self.predictor_ready = True
            self.predictor_error = ""
        except ModuleNotFoundError as err:
            self.model = None
            self.predictor_ready = False
            if "xgboost" in str(err):
                self.predictor_error = "Thiếu thư viện bắt buộc 'xgboost' để tải mô hình dự báo."
            else:
                self.predictor_error = f"Thiếu thư viện khi tải model: {err}"
        except FileNotFoundError as err:
            self.model = None
            self.predictor_ready = False
            self.predictor_error = f"Không tìm thấy file model: {err}"
        except Exception as err:
            self.model = None
            self.predictor_ready = False
            self.predictor_error = f"Lỗi tải model dự báo: {err}"

        self.model_ready = self.predictor_ready
        self.model_error = self.predictor_error

        self._update_runtime_data(self.df_new)

    def _update_shap_readiness(self):
        if importlib.util.find_spec("shap") is None:
            self.shap_ready = False
            self.shap_error = "Thiếu thư viện 'shap' để giải thích mô hình."
            return

        self.shap_ready = True
        self.shap_error = ""

    def _update_runtime_data(self, dataframe):
        self.df_new = dataframe.copy()
        self.X_scaled = None
        self.churn_probs = None

        if self.scaler is None:
            return

        self.X_scaled = self.scaler.transform(self.df_new)
        if self.predictor_ready and self.model is not None:
            self.churn_probs = self.model.predict_proba(self.X_scaled)[:, 1]

    def import_csv(self, file_path):
        if not file_path or not str(file_path).lower().endswith(".csv"):
            raise ValueError("File không hợp lệ. Vui lòng chọn file .csv.")

        read_error = None
        df_imported = None
        for encoding in ["utf-8-sig", "utf-8", "cp1258", "latin-1"]:
            try:
                df_imported = pd.read_csv(file_path, encoding=encoding)
                read_error = None
                break
            except UnicodeDecodeError as err:
                read_error = err
            except Exception as err:
                read_error = err
                break

        if df_imported is None:
            raise ValueError(f"Không thể đọc file CSV. Chi tiết: {read_error}")
        if df_imported.empty:
            raise ValueError("File CSV rỗng, không có dữ liệu để nạp.")

        imported_columns = list(df_imported.columns)
        missing_columns = [col for col in self.expected_columns if col not in imported_columns]
        extra_columns = [col for col in imported_columns if col not in self.expected_columns]
        if missing_columns or extra_columns:
            messages = []
            if missing_columns:
                messages.append(f"Thiếu cột: {', '.join(missing_columns)}")
            if extra_columns:
                messages.append(f"Cột không mong đợi: {', '.join(extra_columns)}")
            raise ValueError("Schema CSV không hợp lệ. " + " | ".join(messages))

        df_imported = df_imported[self.expected_columns]
        if df_imported.isnull().values.any():
            raise ValueError("CSV có dữ liệu rỗng (NaN). Vui lòng làm sạch trước khi import.")

        try:
            self._update_runtime_data(df_imported)
        except Exception as err:
            raise ValueError(f"Dữ liệu CSV không tương thích với scaler/model: {err}")

        return len(df_imported)

    def get_display_dataframe(self):
        df_display = self.df_new.copy()
        if self.churn_probs is not None:
            df_display['Risk_Score (%)'] = (self.churn_probs * 100).round(2)
            return df_display.sort_values(by='Risk_Score (%)', ascending=False)

        df_display['Risk_Score (%)'] = "N/A"
        return df_display

    def get_kpis(self, df_display_sorted):
        total_customers = len(df_display_sorted)
        risk_series = pd.to_numeric(df_display_sorted['Risk_Score (%)'], errors='coerce')
        if risk_series.notna().any():
            avg_risk = float(risk_series.mean())
            high_risk_count = int((risk_series > 70).sum())
        else:
            avg_risk = 0.0
            high_risk_count = 0
        return total_customers, avg_risk, high_risk_count

    def get_shap_figure(self, original_index):
        if not self.predictor_ready or self.model is None or self.X_scaled is None:
            detail = self.predictor_error if self.predictor_error else "Model dự báo chưa được khởi tạo."
            raise RuntimeError(f"Không thể phân tích vì model dự báo chưa sẵn sàng. Chi tiết: {detail}")
        if not self.shap_ready:
            raise RuntimeError(f"Không thể phân tích SHAP. Chi tiết: {self.shap_error}")

        row_idx = self.df_new.index.get_loc(original_index)
        customer_data = self.X_scaled[row_idx].reshape(1, -1)
        import shap

        explainer = shap.Explainer(self.model)
        shap_values_single = explainer(customer_data)

        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(shap_values_single[0], max_display=10, show=False)
        plt.tight_layout()
        return fig