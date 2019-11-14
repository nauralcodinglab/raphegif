import unittest

import numpy as np

import Tools


# SPECIFY TESTS

class Test_check_dict_fields(unittest.TestCase):
    """Test behaviour of `check_dict_fields` function."""

    def test_non_recursive(self):
        """`check_dict_fields` behaviour for shallow dicts."""
        template = {'A': None, 'sometimes_missing': None, 'field_name': None}
        try:
            Tools.check_dict_fields(
                {
                    'A': 1,
                    '1': 2,
                    'sometimes_missing': 'content',
                    'field_name': {'key1': None, 'key2': 'str'}
                },
                template
            )
        except:
            self.fail('`check_dict_fields` raised an exception unexpectedly.')

        with self.assertRaises(KeyError) as context:
            Tools.check_dict_fields(
                {'A': 1, 'field_name': {'key1': None, 'key2': 'str'}},
                template
            )
        self.assertTrue(
            'sometimes_missing' in str(context.exception),
            'Missing key `sometimes_missing` not in exception message {}.'.format(
                context.exception
            )
        )

    def test_recursive(self):
        """`check_dict_fields` behaviour for deep dicts."""
        template = {
            'A': None,
            'outer_field': {
                'inner_field1': {'innermost_field': None},
                'inner_field2': None
            }
        }
        # Test successful check.
        try:
            Tools.check_dict_fields(
                {
                    'A': 1.,
                    'outer_field': {
                        'inner_field1': {'innermost_field': 2.},
                        'inner_field2': 'some_content'
                    }
                },
                template
            )
        except:
            self.fail('`check_dict_fields` raised an exception unexpectedly.')

        # Test unsuccessful check due to missing field.
        with self.assertRaises(KeyError) as context:
            Tools.check_dict_fields(
                {
                    'A': 1.,
                    'outer_field': {
                        'inner_field1': {'wrong_field': 2.},
                        'inner_field2': 'some_content'
                    }
                },
                template
            )
        self.assertTrue(
            'outer_field/inner_field1/innermost_field' in str(context.exception),
            'Missing key name not in exception message.'
        )

        # Test unsuccessful check due to non-subscriptable field in x.
        with self.assertRaises(KeyError) as context:
            Tools.check_dict_fields(
                {
                    'A': 1.,
                    'outer_field': {
                        'inner_field1': 'non subscriptable',
                        'inner_field2': 'some_content'
                    }
                },
                template
            )
        self.assertTrue(
            'outer_field/inner_field1/innermost_field' in str(context.exception),
            'Missing key name not in exception message.'
        )


class Test_validate_array_ndim(unittest.TestCase):

    def test_passes_correct_ndim(self):
        for required_ndim in [1, 2, 5]:
            try:
                Tools.validate_array_ndim(
                    'label',
                    np.empty([5 for i in range(required_ndim)]),
                    required_ndim
                )
            except ValueError as e:
                self.fail(
                    'Raised ValueError `{}` unexpectedly for {} '
                    'dimensions.'.format(e.message, required_ndim)
                )

    def test_raises_error_for_incorrect_ndim(self):
        required_ndim = 3
        for passed_ndim in [1, 2, 5]:
            with self.assertRaises(ValueError):
                Tools.validate_array_ndim(
                    'label',
                    np.empty([5 for i in range(passed_ndim)]),
                    required_ndim
                )


class Test_validate_matching_axis_lengths(unittest.TestCase):

    def test_passes_matching_dimensions(self):
        try:
            Tools.validate_matching_axis_lengths(
                [np.empty((2, 10, 3)), np.empty((2, 5, 3)), np.empty((2, 3, 3))],
                [0, 2]
            )
        except ValueError as e:
            self.fail(
                'Raised ValueError `{}` unexpectedly for matching along '
                'two of three dimensions.'.format(e.message)
            )

    def test_raises_error_for_non_matching_lengths(self):
        with self.assertRaises(ValueError) as error_catcher:
            Tools.validate_matching_axis_lengths(
                [np.empty((2, 10, 3)), np.empty((2, 5, 3)), np.empty((2, 3, 3))],
                [0, 1]  # Note: axis 1 is non-matching.
            )
        self.assertTrue(
            'along axis 1' in str(error_catcher.exception),
            'Non-matching axis 1 not stated correctly in error message.'
        )



if __name__ == '__main__':
    unittest.main()
