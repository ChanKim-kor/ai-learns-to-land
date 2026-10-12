import unittest

from physics import free_fall_velocity, simulate_free_fall, simulate_constant_thrust

class TestFreeFall(unittest.TestCase):
    def test_velocity_after_two_seconds(self):
        # Calculate the actual velocity and check the expected value here.
        velocity = -10
        g = 9.81
        elapsed = 2
        actual = free_fall_velocity(velocity, g, elapsed)
        expected = -29.62
        self.assertAlmostEqual(actual, expected, places=6)

    def test_time_limit_before_ground_contact(self):
        height = 100
        velocity = -10
        g = 9.81
        dt = 0.1
        max_time = 2
        result = simulate_free_fall(height, velocity, g, dt, max_time)
        self.assertAlmostEqual(result["history"][-1][0], 2, places=6)
        self.assertEqual(result["termination_reason"], "time_limit")

    def test_hover_with_balanced_thrust(self):
        height = 100
        velocity = 0
        mass = 100
        g = 9.81
        thrust = 981
        dt = 0.1
        max_time = 2

        result = simulate_constant_thrust(height, velocity, thrust, mass, g, dt, max_time)
        last_time, last_height, last_velocity = result["history"][-1]
        self.assertAlmostEqual(last_time, 2, places=6)
        self.assertAlmostEqual(last_height, height, places=6)
        self.assertAlmostEqual(last_velocity, velocity, places=6)
        self.assertEqual(result["termination_reason"], "time_limit")

    def test_simulation_rejects_zero_time_step(self):
        height = 100
        velocity = -10
        g = 9.81
        dt = 0
        with self.assertRaises(ValueError):
            simulate_free_fall(height, velocity, g, dt)

    def test_simulation_stops_at_exact_ground_contact(self):
        height = 100
        velocity = -10
        g = 9.81
        dt = 1
        max_time = 20
        result = simulate_free_fall(height, velocity, g, dt, max_time)

        last_time, last_height, last_velocity = result["history"][-1]

        self.assertAlmostEqual(last_time, 3.609505622728941, places=6)
        self.assertAlmostEqual(last_height, 0, places=6)
        self.assertAlmostEqual(last_velocity, -45.40925015897091, places=6)
        self.assertEqual(result["termination_reason"], "ground_contact")

        

if __name__ == "__main__":
    unittest.main()