import unittest

from landing import classify_landing

class TestLanding(unittest.TestCase):
    def test_no_contact_is_in_flight(self):
        landing_status = classify_landing(velocity=-20, ground_contact=False)
        self.assertEqual(landing_status, "in_flight")

    def test_contact_at_speed_limit_is_landed(self):
        landing_status = classify_landing(velocity=-2, ground_contact=True)
        self.assertEqual(landing_status, "landed")

    def test_contact_above_speed_limit_is_crached(self):
        landing_status = classify_landing(velocity=-2.01, ground_contact=True)
        self.assertEqual(landing_status, "crashed")

if __name__ == "__main__":
    unittest.main()