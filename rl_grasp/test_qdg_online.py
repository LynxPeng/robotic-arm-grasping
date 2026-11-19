import json
from pathlib import Path
import tempfile
import unittest

from qdg_env import load_manifest


class QDGOnlineTest(unittest.TestCase):
    def test_splits_reject_duplicate_mesh_content(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in ('train', 'holdout'):
                (folder / f'{name}.obj').write_text('v 0 0 0\n', encoding='ascii')
                entry = dict(object=name, mesh=f'{name}.obj', mass=.05,
                             bounds=[[-.02, -.02, -.02], [.02, .02, .02]],
                             inertia_origin=[0, 0, 0, 0, 0, 0, 1])
                (folder / f'{name}.json').write_text(json.dumps([entry]), encoding='utf-8')
            self.assertEqual(tuple(load_manifest(folder, 'train')), ('train',))
            from qdg_env import dataset_signature
            with self.assertRaisesRegex(ValueError, 'same mesh'):
                dataset_signature(folder)
