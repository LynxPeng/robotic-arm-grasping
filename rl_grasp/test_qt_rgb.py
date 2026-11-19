import unittest

import numpy as np
from qt_rgb_env import RGBGraspEnv, OBJECTS, expert


class RGBGraspTest(unittest.TestCase):
    def test_sensor_contract_and_scripted_reachability(self):
        env = RGBGraspEnv(seed=12, max_steps=100)
        try:
            for kind in OBJECTS:
                with self.subTest(kind=kind):
                    image, aux = env.reset(seed=77, kind=kind)
                    self.assertEqual(image.shape, (224, 224, 3))
                    self.assertEqual(image.dtype, np.uint8)
                    self.assertEqual(aux.shape, (4,))
                    phase = wait = 0
                    for _ in range(env.max_steps):
                        action, phase, wait = expert(env, phase, wait)
                        self.assertEqual(len(action), 5)
                        _, _, done, timeout, info = env.step(action)
                        if done or timeout:
                            break
                    self.assertTrue(info['success'], info)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
