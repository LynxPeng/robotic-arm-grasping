import unittest

import torch

from qt_model import cem_action, td_target


class QTModelTest(unittest.TestCase):
    def test_cem_tracks_distinct_optima_per_observation(self):
        torch.manual_seed(17)
        goals = torch.tensor([[0.6, -0.35, 0.2, -1.0], [-0.5, 0.35, -0.2, 1.0]])

        def score(obs, action):
            return -(action - obs[:, None]).square().sum(-1)

        actions = cem_action(score, goals, population=384, iterations=4)
        torch.testing.assert_close(actions, goals, rtol=0, atol=0.12)

    def test_cem_can_limit_five_dimensional_search_around_a_learned_center(self):
        torch.manual_seed(41)
        observation = torch.zeros(2, 1)
        center = torch.tensor([[.2, -.1, .45, -.5, .6], [-.1, .4, -.2, .7, -.5]])
        radius = torch.tensor([.12, .12, .12, .20, .14])

        def score(obs, actions):
            return actions[..., -1]

        action = cem_action(score, observation, population=192, iterations=4,
                            action_dim=5, center=center, radius=radius)
        self.assertTrue(torch.all((action-center).abs() <= radius + 1e-5))
        torch.testing.assert_close(action[:, -1], center[:, -1]+radius[-1], atol=.03, rtol=0)

    def test_terminal_transitions_do_not_bootstrap(self):
        actual = td_target(torch.tensor([1., .2]), torch.tensor([True, False]),
                           torch.tensor([9., .7]), .9)
        torch.testing.assert_close(actual, torch.tensor([1., .83]))


if __name__ == '__main__':
    unittest.main()
