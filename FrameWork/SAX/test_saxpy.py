from saxpy import SAX
import numpy as np


class TestSAX(object):
    def setUp(self):
        # All tests will be run with 6 letter words
        # and 5 letter alphabet
        self.sax = SAX(16, 4, 1e-6)

    def test_long_to_letter_rep(self):
        long_arr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 23.0, 73.0, 73.0, 75.0, 30.0, 16.0, 19.0, 27.0, 33.0, 19.0, 5.0, 20.0, 19.0, 13.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 18.0, 16.0, 11.0, 30.0, 10.0, 39.0, 12.0, 2.0, 15.0, 16.0, 4.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 19.0, 6.0, 39.0, 27.0, 18.0, 20.0, 38.0, 34.0, 33.0, 10.0, 10.0, 15.0, 10.0, 8.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 9.0, 10.0, 10.0, 35.0, 25.0, 24.0, 18.0, 28.0, 18.0, 16.0, 18.0, 31.0, 10.0, 10.0, 15.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 8.0, 30.0, 25.0, 13.0, 13.0, 28.0, 27.0, 20.0, 13.0, 9.0, 11.0, 5.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 11.0, 4.0, 18.0, 26.0, 13.0, 23.0, 16.0, 13.0, 15.0, 12.0, 17.0, 15.0, 24.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.0, 0.0, 0.0, 10.0, 9.0, 3.0, 27.0, 15.0, 18.0, 23.0, 25.0, 16.0, 12.0, 23.0, 13.0, 16.0, 10.0, 8.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 13.0, 8.0, 28.0, 25.0, 19.0, 15.0, 23.0, 8.0, 23.0, 30.0, 28.0, 20.0, 25.0, 16.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 16.0, 10.0, 27.0, 24.0, 30.0, 27.0, 28.0, 41.0, 31.0, 25.0, 6.0, 25.0, 9.0, 9.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 11.0, 25.0, 28.0, 15.0, 15.0, 23.0, 15.0, 23.0, 26.0, 15.0, 17.0, 12.0, 9.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 18.0, 12.0, 36.0, 28.0, 13.0, 21.0, 15.0, 19.0, 33.0, 36.0, 9.0, 6.0, 10.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 12.0, 12.0, 42.0, 13.0, 23.0, 23.0, 49.0, 5.0, 6.0, 15.0, 13.0, 13.0, 11.0, 16.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 4.0, 2.0, 16.0, 25.0, 17.0, 16.0, 25.0, 18.0, 18.0, 25.0, 17.0, 13.0, 12.0, 4.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 11.0, 3.0, 28.0, 20.0, 24.0, 21.0, 21.0, 21.0, 16.0, 32.0, 28.0, 15.0, 18.0, 15.0, 2.0, 11.0, 23.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 17.0, 13.0, 26.0, 15.0, 18.0, 15.0, 3.0, 0.0, 11.0, 19.0, 11.0, 17.0, 12.0, 4.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 18.0, 16.0, 26.0, 15.0, 19.0, 18.0, 20.0, 26.0, 11.0, 12.0, 10.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 12.0, 10.0, 45.0, 20.0, 15.0, 28.0, 20.0, 24.0, 16.0, 19.0, 20.0, 13.0, 19.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 9.0, 4.0, 11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 2.0, 5.0, 1.0, 15.0, 8.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 15.0, 12.0, 21.0, 25.0, 15.0, 15.0, 26.0, 2.0, 0.0, 2.0, 0.0, 4.0, 12.0, 16.0, 18.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.0, 0.0, 10.0, 12.0, 6.0, 20.0, 0.0, 0.0, 1.0, 27.0, 19.0, 25.0, 3.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 24.0, 11.0, 25.0, 17.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 16.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 26.0, 26.0, 17.0, 6.0, 18.0, 17.0, 8.0, 17.0, 4.0, 21.0, 12.0, 16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 0.0, 6.0, 5.0, 16.0, 18.0, 23.0, 32.0, 17.0, 25.0, 5.0, 12.0, 13.0, 0.0, 0.0, 0.0]
        print(len(long_arr))
        (letters, indices) = self.sax.to_letter_rep(long_arr)

        print(letters)
        (letters, indices) = self.sax.to_letter_rep(long_arr[100:])
        print(letters)
        (letters, indices) = self.sax.to_letter_rep(long_arr[:100]+long_arr[200:])
        print(letters)

        # assert letters == 'bbbbce'



a = TestSAX()
a.setUp()
a.test_long_to_letter_rep()