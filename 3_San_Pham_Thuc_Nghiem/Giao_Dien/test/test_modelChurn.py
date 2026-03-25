import unittest
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
giao_dien_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(giao_dien_dir)

from model.modelChurn import ChurnModel

class TestChurnModel(unittest.TestCase):
    def setUp(self):
        # Trỏ đường dẫn test tới thư mục chứa file đã xử lý thực tế
        data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'Du_Lieu', 'Da_Xu_Ly'))
        self.model = ChurnModel(data_dir=data_dir)

    def test_dataframe_shape(self):
        df = self.model.get_display_dataframe()
        self.assertTrue('Risk_Score (%)' in df.columns)
        self.assertGreater(len(df), 0)

    def test_shap_figure(self):
        df = self.model.get_display_dataframe()
        first_index = df.index[0]
        fig = self.model.get_shap_figure(first_index)
        self.assertIsNotNone(fig)

if __name__ == '__main__':
    unittest.main()