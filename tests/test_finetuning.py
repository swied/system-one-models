"""Offline tests for split integrity, calibration, and checkpoint adapter contracts."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from system_one_models.finetuning import (
    CheckpointBackend, encode_examples, fit_temperature, scenarios, verify_dataset,
)

spec = importlib.util.spec_from_file_location('ticket_data', Path(__file__).parents[1] / 'week_03/ticket_data.py')
ticket_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ticket_data)


class DataTests(unittest.TestCase):
    def test_frozen_balanced_disjoint_splits_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            ticket_data.generate(a, 600)
            ticket_data.generate(b, 600)
            splits = verify_dataset(a)
            self.assertEqual({s: len(r) for s, r in splits.items()},
                             {'train': 480, 'validation': 30, 'calibration': 30, 'test': 60})
            for name in splits:
                self.assertEqual((Path(a) / f'{name}.jsonl').read_bytes(),
                                 (Path(b) / f'{name}.jsonl').read_bytes())
            scenario = scenarios(splits['train'][:1])[0]
            self.assertIn(scenario['labels']['team'], scenario['questions']['team']['criteria'])
            with self.assertRaises(ValueError):
                ticket_data.generate(a, 600)
            with (Path(a) / 'test.jsonl').open('a') as handle:
                handle.write('\n')
            with self.assertRaisesRegex(ValueError, 'Frozen split changed'):
                verify_dataset(a)

    def test_cross_family_leakage_detected_even_with_updated_digest(self):
        from system_one_models.finetuning import file_digest
        with tempfile.TemporaryDirectory() as d:
            ticket_data.generate(d, 120)
            splits = verify_dataset(d)
            path = Path(d) / 'test.jsonl'
            rows = splits['test']
            rows[0]['family'] = splits['train'][0]['family']
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            manifest_path = Path(d) / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['sha256']['test'] = file_digest(path)
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'leakage'):
                verify_dataset(d)


class CalibrationTests(unittest.TestCase):
    def test_temperature_reduces_overconfidence_without_changing_choices(self):
        from scipy.special import logsumexp
        z = np.array([[8., 0., 0.]] * 10)
        y = np.array([0] * 6 + [1] * 2 + [2] * 2)
        t = fit_temperature(z, y)
        nll = lambda x: np.mean(logsumexp(x, axis=1) - x[np.arange(len(y)), y])
        self.assertGreater(t, 1)
        self.assertLess(nll(z / t), nll(z))
        np.testing.assert_array_equal(z.argmax(1), (z / t).argmax(1))

    def test_invalid_calibration(self):
        for z, y in (([], []), ([[float('nan'), 0]], [0]), ([[0, 1]], [2])):
            with self.assertRaises(ValueError):
                fit_temperature(z, y)


class EncodingTests(unittest.TestCase):
    class Tokenizer:
        mask_token = '[MASK]'
        mask_token_id = 3
        cls_token_id = 1
        sep_token_id = 2
        pad_token_id = 0
        def encode(self, text, add_special_tokens=False, **kwargs):
            return [sum((i + 1) * ord(c) for i, c in enumerate(word)) + 5
                    for word in text.split()[:kwargs.get('max_length', 100000)]]
        def __call__(self, text, **kwargs):
            return {'input_ids': self.encode(text, **kwargs)}

    def agent(self):
        from types import SimpleNamespace
        return SimpleNamespace(tok=self.Tokenizer(), cfg={'max_len': 512, 'head_max_len': 192})

    def row(self):
        return {'id': 'x', 'state': 'Please refund my payment.', 'question': 'Which team?',
                'options': ticket_data.OPTIONS, 'label': 'billing'}

    def test_encoding_and_token_limits(self):
        row = self.row()
        items = encode_examples(self.agent(), [row])
        self.assertEqual(items[0]['target'], [1., 0., 0.])
        row['options'] = {'a': 'word ' * 60, 'b': 'short'}
        row['label'] = 'a'
        with self.assertRaisesRegex(ValueError, '48 tokens'):
            encode_examples(self.agent(), [row])
        row = self.row()
        row['state'] = 'word ' * 600
        with self.assertRaisesRegex(ValueError, 'truncate'):
            encode_examples(self.agent(), [row])
        row = self.row()
        row['question'] = 'word ' * 200
        with self.assertRaisesRegex(ValueError, 'head would truncate'):
            encode_examples(self.agent(), [row])

    def test_checkpoint_response_contract(self):
        from types import SimpleNamespace
        agent = self.agent()
        agent.device = SimpleNamespace(type='cpu')
        agent.system_one = lambda **kwargs: {'answers': {'team': {'choice': 'billing'}}}
        with patch('laya.load', return_value=agent):
            backend = CheckpointBackend('tuned', '/tmp/checkpoint', device='cpu')
            self.assertEqual(backend.predict('refund', scenarios([self.row()])[0]['questions']),
                             {'team': {'choice': 'billing'}})


class ObjectiveTests(unittest.TestCase):
    def test_exploration_loss_has_finite_learning_signal(self):
        import torch
        train_spec = importlib.util.spec_from_file_location(
            'week03_train', Path(__file__).parents[1] / 'week_03/train.py')
        module = importlib.util.module_from_spec(train_spec)
        train_spec.loader.exec_module(module)
        torch.manual_seed(42)
        logits = torch.tensor([[0.2, -0.1, 0.3]], requires_grad=True)
        loss = module.rlcd_loss(logits, torch.tensor([[1., 0., 0.]]), 0.4, 4)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0)
        with self.assertRaises(ValueError):
            module.rlcd_loss(logits, torch.tensor([[1., 0., 0.]]), 0.0, 4)


if __name__ == '__main__':
    unittest.main()
