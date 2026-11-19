"""Compare CUDA Graph and ordinary TD3 updates on identical CUDA batches."""
import torch
from copy import deepcopy

from ga_env import ACT_DIM, OBS_DIM
from ga_model import Learner


def compare_updates(device, auxiliary_weight, cuda_graph=True):
    torch.manual_seed(41)
    eager = Learner(device, 1e-4, auxiliary_weight)
    torch.manual_seed(41)
    graph = Learner(device, 1e-4, auxiliary_weight, cuda_graph=cuda_graph)
    size = 16
    max_loss_diff = 0.
    for index in range(12):
        # Compare each update from exactly the same state, so atomic rounding
        # cannot compound through later max-pool winner changes. Only the eager
        # reference is reloaded; captured graph storage is never replaced.
        torch.cuda.synchronize()
        for name in ('actor', 'critic', 'actor_target', 'critic_target', 'actor_opt', 'critic_opt'):
            getattr(eager, name).load_state_dict(deepcopy(getattr(graph, name).state_dict()))
        actions = torch.rand(size, ACT_DIM, device=device) * 2 - 1
        labels = torch.rand(size, ACT_DIM, device=device) * 2 - 1
        batch = (
            torch.randn(size, OBS_DIM, device=device), actions, labels,
            torch.randint(0, 2, (size,), device=device).float(),
            torch.randn(size, OBS_DIM, device=device),
            torch.randint(0, 2, (size,), device=device).float(),
            torch.randn(size, 6, device=device),
            torch.randint(0, 2, (size, 2), device=device).float(),
        )
        noise = torch.randn_like(actions)
        expected = eager.update(batch, target_noise=noise)
        actual = graph.update(batch, target_noise=noise)
        assert ('bc_loss' in actual) == (index % 2 == 1)
        max_loss_diff = max(max_loss_diff, max((actual[k]-expected[k]).abs().item() for k in actual))
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=1e-4)
    assert graph.updates == eager.updates == 12
    assert graph.graph_replays == (8 if cuda_graph else 0)
    maximum = 0.
    for name in ('actor', 'critic', 'actor_target', 'critic_target'):
        for key, value in getattr(graph, name).state_dict().items():
            expected = getattr(eager, name).state_dict()[key]
            maximum = max(maximum, (value-expected).abs().max().item())
            if 'num_batches_tracked' in key:
                torch.testing.assert_close(value, expected, atol=0, rtol=0)
                assert value.item() == (6 if name.startswith('actor') else 12)
        # Upstream grouping backward uses atomicAdd: independent eager runs
        # are also non-bitwise-deterministic. Test eager/eager as a control.
        torch.testing.assert_close(getattr(graph, name).state_dict(),
                                   getattr(eager, name).state_dict(), atol=3e-6, rtol=1e-4)
    for name, steps in (('actor_opt', 6), ('critic_opt', 12)):
        actual = getattr(graph, name).state_dict()['state']
        expected = getattr(eager, name).state_dict()['state']
        assert all(state['step'].item() == steps for state in actual.values())
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=1e-4)
    print(f'cuda_graph={cuda_graph} aux={auxiliary_weight} max_state_abs_diff={maximum:.8g} '
          f'max_loss_abs_diff={max_loss_diff:.8g}', flush=True)


if __name__ == '__main__':
    device = torch.device('cuda')
    for weight in (0., 0.5):
        compare_updates(device, weight, cuda_graph=False)
        compare_updates(device, weight)
        print(f'CUDA Graph matches ordinary updates with aux_weight={weight}')
