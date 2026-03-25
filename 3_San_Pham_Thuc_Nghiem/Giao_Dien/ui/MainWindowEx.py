from PyQt6.QtWidgets import (
    QMainWindow,
    QTableWidgetItem,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox
)
from PyQt6.QtGui import QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sys
import os
import traceback

current_dir = os.path.dirname(os.path.abspath(__file__)) # Lấy đường dẫn thư mục ui/
parent_dir = os.path.abspath(os.path.join(current_dir, '..')) # Lùi lại 1 cấp ra Giao_Dien/

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from model.modelChurn import ChurnModel
from ui.MainWindow import Ui_MainWindow

class MainWindowEx(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self._setup_csv_import_ui()
        self.model_churn = None
        self.df_display = None
        # Cấu hình UI
        self.tableWidget.setEditTriggers(self.tableWidget.EditTrigger.NoEditTriggers)
        self.setup_connections()

        # Load Model và Data
        try:
            self.model_churn = ChurnModel()
            self.df_display = self.model_churn.get_display_dataframe()
            self.setup_data()
            if not self.model_churn.predictor_ready:
                self.labelCsvStatus.setText(
                    f"Model dự báo chưa sẵn sàng: {self.model_churn.predictor_error}"
                )
            elif not self.model_churn.shap_ready:
                self.labelCsvStatus.setText(
                    f"SHAP chưa sẵn sàng: {self.model_churn.shap_error}"
                )
        except Exception as e:
            traceback.print_exc()
            self.labelTitle.setText(f"❌ Lỗi tải dữ liệu: {str(e)}")
            self.labelCsvStatus.setText("Không thể nạp model ban đầu")
            self._update_analysis_ui_state()

    def setup_data(self):
        # 1. Cập nhật KPI
        total, avg_risk, high_risk = self.model_churn.get_kpis(self.df_display)
        self.labelKPI1.setText(f"Tổng khách hàng: {total}")
        self.labelKPI2.setText(f"Trung bình rủi ro: {avg_risk:.2f}%")
        self.labelKPI3.setText(f"Khách hàng rủi ro cao (>70%): {high_risk}")

        # 2. Đổ dữ liệu vào TableWidget
        headers = ["Index"] + list(self.df_display.columns)
        self.tableWidget.setColumnCount(len(headers))
        self.tableWidget.setHorizontalHeaderLabels(headers)
        self.tableWidget.setRowCount(len(self.df_display))

        for row_idx, (index, row) in enumerate(self.df_display.iterrows()):
            # Cột Index
            self.tableWidget.setItem(row_idx, 0, QTableWidgetItem(str(index)))

            # Các cột data
            for col_idx, col_name in enumerate(self.df_display.columns):
                val = row[col_name]
                item = QTableWidgetItem(str(val))

                # Tô màu cột Risk Score
                if col_name == 'Risk_Score (%)':
                    if isinstance(val, (int, float)):
                        if val > 70:
                            item.setBackground(QColor(255, 200, 200))  # Đỏ nhạt
                        elif val > 40:
                            item.setBackground(QColor(255, 230, 200))  # Cam nhạt
                        else:
                            item.setBackground(QColor(200, 255, 200))  # Xanh nhạt

                self.tableWidget.setItem(row_idx, col_idx + 1, item)

        # 3. Đổ dữ liệu vào ComboBox
        self.comboBoxCustomer.clear()
        self.comboBoxCustomer.addItems([str(idx) for idx in self.df_display.index.tolist()])
        self._update_analysis_ui_state()

    def _update_analysis_ui_state(self):
        if self.model_churn is None:
            self.btnAnalyze.setEnabled(False)
            self.btnAnalyze.setToolTip("Bộ xử lý dữ liệu chưa sẵn sàng.")
            return

        if not self.model_churn.predictor_ready:
            self.btnAnalyze.setEnabled(False)
            self.btnAnalyze.setToolTip(
                f"Model dự báo chưa sẵn sàng: {self.model_churn.predictor_error}"
            )
            return

        if not self.model_churn.shap_ready:
            self.btnAnalyze.setEnabled(False)
            self.btnAnalyze.setToolTip(
                f"SHAP chưa sẵn sàng: {self.model_churn.shap_error}"
            )
            return

        self.btnAnalyze.setEnabled(True)
        self.btnAnalyze.setToolTip("")

    def setup_connections(self):
        self.btnAnalyze.clicked.connect(self.plot_shap)
        self.btnImportCsv.clicked.connect(self.import_csv_file)

    def _setup_csv_import_ui(self):
        self.btnImportCsv = QPushButton("📂 Thêm file CSV", self.centralwidget)
        self.btnImportCsv.setObjectName("btnImportCsv")
        self.horizontalLayout_Action.addWidget(self.btnImportCsv)

        self.labelCsvStatus = QLabel("Chưa chọn file CSV", self.centralwidget)
        self.labelCsvStatus.setObjectName("labelCsvStatus")
        self.horizontalLayout_Action.addWidget(self.labelCsvStatus)

    def import_csv_file(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Chọn file CSV",
                "",
                "CSV Files (*.csv)"
            )
        except Exception as err:
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "Lỗi mở hộp thoại",
                f"Không thể mở cửa sổ chọn file CSV.\nChi tiết: {err}"
            )
            self.labelCsvStatus.setText("Lỗi mở hộp thoại chọn file")
            return

        if not file_path:
            self.labelCsvStatus.setText("Đã hủy chọn file CSV")
            return

        if not file_path.lower().endswith(".csv"):
            QMessageBox.critical(self, "Lỗi file", "Chỉ chấp nhận file .csv")
            self.labelCsvStatus.setText("File không hợp lệ")
            return

        try:
            if self.model_churn is None:
                raise RuntimeError("Không thể khởi tạo bộ xử lý dữ liệu.")

            row_count = self.model_churn.import_csv(file_path)
            self.df_display = self.model_churn.get_display_dataframe()
            self.setup_data()

            file_name = os.path.basename(file_path)
            self.labelCsvStatus.setText(f"Đã nạp: {file_name} ({row_count} dòng)")
            if self.model_churn.predictor_ready:
                QMessageBox.information(
                    self,
                    "Import CSV thành công",
                    f"Nạp dữ liệu thành công từ file:\n{file_name}\nSố dòng: {row_count}"
                )
            else:
                QMessageBox.warning(
                    self,
                    "Import CSV thành công (chưa dự báo)",
                    (
                        f"Đã nạp dữ liệu từ file:\n{file_name}\nSố dòng: {row_count}\n\n"
                        "Chưa thể tính Risk Score do model dự báo chưa sẵn sàng.\n"
                        f"Chi tiết: {self.model_churn.predictor_error}"
                    )
                )
        except Exception as err:
            traceback.print_exc()
            self.labelCsvStatus.setText("Import thất bại")
            QMessageBox.critical(self, "Import CSV thất bại", str(err))

    def plot_shap(self):
        if self.model_churn is None:
            QMessageBox.critical(self, "Phân tích thất bại", "Bộ xử lý dữ liệu chưa sẵn sàng.")
            return
        if not self.model_churn.predictor_ready:
            QMessageBox.critical(
                self,
                "Phân tích thất bại",
                (
                    "Không thể phân tích vì model dự báo chưa sẵn sàng.\n"
                    f"Chi tiết: {self.model_churn.predictor_error}"
                )
            )
            return
        if not self.model_churn.shap_ready:
            QMessageBox.critical(
                self,
                "Phân tích thất bại",
                f"Không thể phân tích SHAP.\nChi tiết: {self.model_churn.shap_error}"
            )
            return

        if self.comboBoxCustomer.count() == 0:
            QMessageBox.warning(self, "Thiếu dữ liệu", "Không có dữ liệu để phân tích.")
            return

        selected_idx = int(self.comboBoxCustomer.currentText())
        self.btnAnalyze.setText("Đang phân tích...")
        self.btnAnalyze.setEnabled(False)

        try:
            # Xóa đồ thị cũ nếu có
            for i in reversed(range(self.plotLayout.count())):
                widgetToRemove = self.plotLayout.itemAt(i).widget()
                self.plotLayout.removeWidget(widgetToRemove)
                widgetToRemove.setParent(None)

            # Tạo và hiển thị đồ thị mới
            fig = self.model_churn.get_shap_figure(selected_idx)
            canvas = FigureCanvas(fig)
            self.plotLayout.addWidget(canvas)
        except Exception as err:
            traceback.print_exc()
            QMessageBox.critical(self, "Phân tích thất bại", str(err))
        finally:
            self.btnAnalyze.setText("🔍 Phân tích nguyên nhân")
            self.btnAnalyze.setEnabled(True)