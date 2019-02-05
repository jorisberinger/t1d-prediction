from unittest import TestCase
from data import readData, convertData
import pandas as pd
import numpy as np
filename = "../../data/csv/data-o3.csv"
class TestConvertData(TestCase):
    def test_select_columns(self):
        data = readData.read_data(filename)
        self.assertEqual(28, data.shape[1])
        data = convertData.select_columns(data)
        self.assertEqual(7, data.shape[1])

    def test_create_time_index(self):
        data = readData.read_data(filename)
        data = convertData.select_columns(data)
        self.assertEqual(type(data.index[0]), int)
        data = convertData.create_time_index(data)
        self.assertIsInstance(data.index[0], pd._libs.tslib.Timestamp)

    def test_drop_dat_and_time(self):
        data = readData.read_data(filename)
        data = convertData.select_columns(data)
        data = convertData.create_time_index(data)
        self.assertEqual(7, data.shape[1])
        data = convertData.drop_date_and_time(data)
        self.assertEqual(5, data.shape[1])

    def test_convert_glucose(self):
        data = readData.read_data(filename)
        data = convertData.select_columns(data)
        data = convertData.create_time_index(data)
        self.assertIsInstance(data['glucoseAnnotation'][0], str)
        data = convertData.convert_glucose_annotation(data)
        self.assertIsInstance(data['glucoseAnnotation'][0], float)