import unittest

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


if __name__ == '__main__':
    unittest.main()
