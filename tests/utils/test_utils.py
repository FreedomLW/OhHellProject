import unittest
import numpy as np
from rlohhell.utils.utils import (
    init_54_deck,
    init_standard_deck,
    init_short_deck,
    rank2int,
    print_card,
    elegent_form,
    reorganize,
    tournament,
)
import rlohhell

class TestUtils(unittest.TestCase):

    def test_init_standard_deck(self):
        self.assertEqual(len(init_standard_deck()), 52)

    def test_init_54_deck(self):
        self.assertEqual(len(init_54_deck()), 54)

    def test_init_short_deck(self):
        self.assertEqual(len(init_short_deck()), 36)

    def test_rank2int(self):
        self.assertEqual(rank2int('A'), 14)
        self.assertEqual(rank2int(''), -1)
        self.assertEqual(rank2int('6'), 6)
        self.assertEqual(rank2int('9'), 9)
        self.assertEqual(rank2int('T'), 10)
        self.assertEqual(rank2int('J'), 11)
        self.assertEqual(rank2int('Q'), 12)
        self.assertEqual(rank2int('1000'), None)
        self.assertEqual(rank2int('abc123'), None)
        self.assertEqual(rank2int('K'), 13)
        self.assertEqual(rank2int('3'), None)

    def test_print_cards(self):
        self.assertEqual(len(elegent_form('S9')), 2)
        self.assertEqual(len(elegent_form('ST')), 3)

        print_card(None)
        print_card('S9')
        print_card('ST')

if __name__ == '__main__':
    unittest.main()
