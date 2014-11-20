import unittest
import warnings

from gridsim.unit import units
from gridsim.timeseries import TimeSeriesObject
from gridsim.iodata.input import CSVReader


class TestTimeSeries(unittest.TestCase):

    def test_default_load(self):
        time_series = TimeSeriesObject(CSVReader())

        time_series.load('./test/data/datatest_with_header.csv')

        time_series.set_time(2*units.second)
        self.assertEqual(time_series.time, 2*units.second)
        self.assertEqual(time_series.temperature, 18.0)
        self.assertEqual(time_series.humidity, 0.1)

        time_series.set_time(1.2*units.second)
        self.assertEqual(time_series.time, 1.0*units.second)
        self.assertEqual(time_series.temperature, 20.0)
        self.assertEqual(time_series.humidity, 0.25)

        time_series.set_time(1.0*units.second)
        self.assertEqual(time_series.time, 1.0*units.second)
        self.assertEqual(time_series.temperature, 20.0)
        self.assertEqual(time_series.humidity, 0.25)

        time_series.set_time(0.0*units.second)
        self.assertEqual(time_series.time, 0.0*units.second)
        self.assertEqual(time_series.temperature, 20.0)
        self.assertEqual(time_series.humidity, 0.2)

    def test_convert_load(self):
        time_series = TimeSeriesObject(CSVReader())

        time_series.load('./test/data/datatest_with_header.csv',
                         time_converter=lambda t: t*2*units.minute)

        time_series.set_time(4.0*units.minute)
        self.assertEqual(time_series.time, 4.0*units.minute)
        self.assertEqual(time_series.temperature, 18.0)
        self.assertEqual(time_series.humidity, 0.1)

        time_series.set_time(3.2*units.minute)
        self.assertEqual(time_series.time, 2.0*units.minute)
        self.assertEqual(time_series.temperature, 20.0)
        self.assertEqual(time_series.humidity, 0.25)

        time_series.set_time(2.0*units.minute)
        self.assertEqual(time_series.time, 2.0*units.minute)
        self.assertEqual(time_series.temperature, 20.0)
        self.assertEqual(time_series.humidity, 0.25)

        time_series.set_time(0.0*units.minute)
        self.assertEqual(time_series.time, 0.0*units.minute)
        self.assertEqual(time_series.temperature, 20.0)
        self.assertEqual(time_series.humidity, 0.2)

    def test_warning_no_header(self):

        time_series = TimeSeriesObject(CSVReader())

        warnings.simplefilter('error', SyntaxWarning)
        with self.assertRaises(SyntaxWarning):
            time_series.load('./test/data/datatest_no_header.csv',
                             time_key='hero',
                             time_converter=lambda t: t*2*units.minute)
        warnings.simplefilter('default', SyntaxWarning)

    def test_no_time_header(self):

        time_series = TimeSeriesObject(CSVReader())

        time_series.load('./test/data/datatest_with_no_time_header.csv',
                         time_key='counter',
                         time_converter=lambda t: t*2*units.minute)

        time_series.set_time(4.0*units.minute)
        self.assertEqual(time_series.counter, 4.0*units.minute)
        self.assertEqual(time_series.temperature, 18.0)
        self.assertEqual(time_series.humidity, 0.1)

    def test_change_time_key(self):

        time_series = TimeSeriesObject(CSVReader())

        time_series.load('./test/data/datatest_with_no_time_header.csv',
                         time_key='counter',
                         time_converter=lambda t: t*2*units.minute)
        time_series.map_attribute('counter', 'minutes')

        time_series.set_time(4.0*units.minute)
        self.assertEqual(time_series.minutes, 4.0*units.minute)
        self.assertEqual(time_series.temperature, 18.0)
        self.assertEqual(time_series.humidity, 0.1)

    def test_no_key(self):

        time_series = TimeSeriesObject(CSVReader())

        time_series.load('./test/data/datatest_with_no_time_header.csv',
                         time_key='counter',
                         time_converter=lambda t: t*2*units.minute)

        self.assertRaises(AttributeError, time_series.map_attribute,
                          'toto', 'titi')

    def test_already_present_key(self):

        time_series = TimeSeriesObject(CSVReader())

        time_series.load('./test/data/datatest_with_no_time_header.csv',
                         time_key='counter',
                         time_converter=lambda t: t*2*units.minute)

        self.assertRaises(AttributeError, time_series.map_attribute,
                          'counter', 'counter')

if __name__ == '__main__':
    unittest.main()
