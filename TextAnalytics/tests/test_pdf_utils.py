import os
import sys
import unittest

from pdfminer.high_level import extract_pages

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(module_path)
from pre_process import pdf_utils


class TestPDFUtils(unittest.TestCase):
    def __init__(self, methodName: str = "pdf_utils_test") -> None:
        super().__init__(methodName)
        SAMPLE_PDF = "../samples/S100S37P_2.pdf"
        self.pages = [*extract_pages(SAMPLE_PDF)]

    def test_retrieve_typical_table(self):
        pass


if __name__ == "__main__":
    unittest.main()
