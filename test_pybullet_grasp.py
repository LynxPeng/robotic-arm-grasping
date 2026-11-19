import math
import unittest

import numpy as np

from pybullet_grasp import (align_grasp_direction, approach_waypoints, blend_grasp_target,
                            grasp_axes, grasp_direction, lift_succeeded, nearest_grasp_frame,
                            pixel_to_world, setup_scene, step_pose_toward, target_has_foreground,
                            wrist_camera)


class GraspGeometryTest(unittest.TestCase):
    def test_center_and_axes_of_overhead_camera(self):
        eye = (0.55, 0, 0.65)
        center = pixel_to_world(111.5, 111.5, 0.60, eye)
        np.testing.assert_allclose(center, (0.55, 0, 0.05))
        self.assertGreater(pixel_to_world(111.5, 130, 0.60, eye)[0], center[0])
        self.assertGreater(pixel_to_world(90, 111.5, 0.60, eye)[1], center[1])

    def test_angle_convention_matches_upstream_grasp_rectangle(self):
        np.testing.assert_allclose(grasp_direction(0), (1, 0), atol=1e-8)
        np.testing.assert_allclose(grasp_direction(math.pi / 2), (0, 1), atol=1e-8)

    def test_requires_lift_bilateral_contact_and_clear_support(self):
        self.assertTrue(lift_succeeded(0.12, {9, 10}, False))
        self.assertFalse(lift_succeeded(0.04, {9, 10}, False))
        self.assertFalse(lift_succeeded(0.12, {9}, False))
        self.assertFalse(lift_succeeded(0.12, {9, 10}, True))

    def test_wrist_yaw_rotates_image_right_in_world(self):
        forward, up = (0, 0, -1), (-1, 0, 0)
        eye = (0.55, 0, 0.65)
        world = pixel_to_world(111.5, 140, 0.60, eye, forward, up)
        self.assertAlmostEqual(world[0], eye[0])
        self.assertGreater(world[1], eye[1])
        np.testing.assert_allclose(grasp_direction(0, forward, up), (0, 1), atol=1e-8)

    def test_scene_starts_with_varied_oblique_wrist_views(self):
        import pybullet as p

        client = p.connect(p.DIRECT)
        self.addCleanup(p.disconnect, client)
        rng = np.random.default_rng(123)
        frames = []
        for _ in range(3):
            _, _, robot = setup_scene(p, False, "cylinder", 0.55, 0, 0, rng)
            eye, forward, up = wrist_camera(p, robot)
            self.assertLess(forward[2], -0.7)
            self.assertGreater(np.linalg.norm(forward[:2]), 0.10)
            self.assertAlmostEqual(np.dot(forward, up), 0, places=6)
            self.assertGreater(eye[2], 0.5)
            self.assertFalse(p.getContactPoints(robot))
            frames.append((eye, up))
        self.assertGreater(np.linalg.norm(frames[0][0] - frames[1][0]), 0.01)
        self.assertGreater(np.linalg.norm(frames[0][1] - frames[1][1]), 0.1)

    def test_tilted_gripper_points_along_its_approach(self):
        axes = grasp_axes((1, 0), 35)
        waypoints = approach_waypoints((0.55, 0, 0.02), axes[:, 2])
        delta = waypoints[-1] - waypoints[0]
        np.testing.assert_allclose(delta / np.linalg.norm(delta), axes[:, 2])
        np.testing.assert_allclose(axes.T @ axes, np.eye(3), atol=1e-8)
        self.assertAlmostEqual(float(np.linalg.det(axes)), 1)
        self.assertAlmostEqual(axes[2, 2], -math.cos(math.radians(35)))
        self.assertAlmostEqual(abs(axes[0, 1]), 1)

    def test_closed_loop_target_blending_aligns_pi_symmetric_grasp_axis(self):
        prev = (np.array((0.55, 0.0, 0.04)), np.array((1.0, 0.0)), 0.08)
        obs = (np.array((0.55, 0.0, 0.04)), np.array((-1.0, 0.0)), 0.08)
        xyz, direction, opening, shift = blend_grasp_target(prev, obs)
        np.testing.assert_allclose(xyz, (0.55, 0.0, 0.04))
        np.testing.assert_allclose(direction, (1.0, 0.0), atol=1e-8)
        self.assertAlmostEqual(opening, 0.08)
        self.assertAlmostEqual(shift, 0.0)
        np.testing.assert_allclose(align_grasp_direction((0.0, -1.0), (0.0, 1.0)), (0.0, 1.0))

    def test_closed_loop_pose_step_is_bounded_and_occupancy_check_works(self):
        from scipy.spatial.transform import Rotation

        q0 = Rotation.from_euler("xyz", (math.pi, 0.0, 0.0)).as_quat()
        q1 = Rotation.from_euler("xyz", (math.pi, 0.0, 1.2)).as_quat()
        next_pos, next_quat = step_pose_toward((0.5, 0.0, 0.5), q0, (0.7, 0.0, 0.5), q1,
                                               max_trans=0.02, max_rot=0.2)
        self.assertAlmostEqual(float(np.linalg.norm(next_pos - np.array((0.5, 0.0, 0.5)))), 0.02, places=6)
        rel_angle = float((Rotation.from_quat(q0).inv() * Rotation.from_quat(next_quat)).magnitude())
        self.assertAlmostEqual(rel_angle, 0.2, places=5)
        pts = np.array(((0.55, 0.0, 0.04), (0.65, 0.0, 0.04)))
        self.assertTrue(target_has_foreground(pts, (0.56, 0.0, 0.04), radius=0.025))
        self.assertFalse(target_has_foreground(pts, (0.60, 0.0, 0.04), radius=0.025))
        _, chosen_q = nearest_grasp_frame((1.0, 0.0), 25.0, q0)
        self.assertEqual(chosen_q.shape, (4,))


if __name__ == "__main__":
    unittest.main()
