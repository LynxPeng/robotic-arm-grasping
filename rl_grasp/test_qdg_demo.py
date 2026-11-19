import math
import unittest

import numpy as np
import pybullet as p

from qdg_demo import grasp_tcp


class QDGFrameTest(unittest.TestCase):
    def test_grasp_position_uses_world_inertia_and_panda_hand_offset(self):
        inertia = ([0.5, 0.1, 0.05], p.getQuaternionFromEuler((0, 0, math.pi / 2)))
        hand = [0.02, 0.03, 0.13, 1, 0, 0, 0, 0.8]
        position, orientation = grasp_tcp(inertia, hand)
        np.testing.assert_allclose(position, [0.47, 0.12, 0.075], atol=1e-6)
        np.testing.assert_allclose(p.getMatrixFromQuaternion(orientation),
                                   [0, 1, 0, 1, 0, 0, 0, 0, -1], atol=1e-6)
