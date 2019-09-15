import unittest
import hy
import scikit_learn
import numpy as np
import numpy.testing as npt


class ScikitLearnTest(unittest.TestCase):
    def test_all_but_last(self):
        self.assertEqual(scikit_learn.all_but_last([1, 2, 3]), [1, 2])

    def test_learning_and_predicting(self):
        self.assertEqual(scikit_learn.learning_and_predicting(), [8])

    def test_model_persistence(self):
        self.assertEqual(scikit_learn.model_persistence(), [0])

    def test_model_persistence_from_file(self):
        self.assertEqual(scikit_learn.model_persistence_from_file(), [0])

    def test_type_casting_32(self):
        self.assertEqual(scikit_learn.type_casting_32().dtype, np.float32)

    def test_type_casting_64(self):
        self.assertEqual(scikit_learn.type_casting_64().dtype, np.float64)

    def test_type_casting_more1(self):
        self.assertEqual(scikit_learn.type_casting_more1(), [0, 0, 0])

    def test_type_casting_more2(self):
        self.assertEqual(scikit_learn.type_casting_more2(), ['setosa', 'setosa', 'setosa'])

    def test_refitting_and_updating_parameters1(self):
        npt.assert_array_equal(scikit_learn.refitting_and_updating_parameters1(), [0, 0, 0, 0, 0])

    def test_refitting_and_updating_parameters2(self):
        npt.assert_array_equal(scikit_learn.refitting_and_updating_parameters2(), [0, 0, 0, 0, 0])

    def test_multiclass_vs_multilabel_fitting1(self):
        npt.assert_array_equal(scikit_learn.multiclass_vs_multilabel_fitting1(), [0, 0, 1, 1, 2])

    def test_multiclass_vs_multilabel_fitting2(self):
        npt.assert_array_equal(scikit_learn.multiclass_vs_multilabel_fitting2(), [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0]])

    def test_multiclass_vs_multilabel_fitting3(self):
        npt.assert_array_equal(scikit_learn.multiclass_vs_multilabel_fitting3(), [[1, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 0, 1, 0, 0], [1, 0, 1, 0, 0]])


if __name__ == "__main__":
  unittest.main()
