"""Compare the final full checkpoints of uninterrupted and resumed runs."""
import argparse
from pathlib import Path

import torch


def compare(actual, expected, key='checkpoint'):
    if isinstance(expected, torch.Tensor):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype, key
        assert torch.equal(actual.contiguous().reshape(-1).view(torch.uint8),
                           expected.contiguous().reshape(-1).view(torch.uint8)), key
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), key
        for name, value in expected.items():
            compare(actual[name], value, f'{key}.{name}')
    elif isinstance(expected, (list, tuple)):
        for i, (value, ref) in enumerate(zip(actual, expected, strict=True)):
            compare(value, ref, f'{key}[{i}]')
    else:
        assert actual == expected, key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('uninterrupted', type=Path)
    parser.add_argument('resumed', type=Path)
    args = parser.parse_args()
    reference = torch.load(args.uninterrupted, map_location='cpu', weights_only=True)
    resumed = torch.load(args.resumed, map_location='cpu', weights_only=True)
    compare(resumed, reference)
    print(f"Full checkpoint matches bitwise at {reference['steps']} environment steps "
          f"and {reference['updates']} updates.")


if __name__ == '__main__':
    main()
