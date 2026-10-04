"""Offline regression checks; run with python -m unittest discover -s tests."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from typesafe_sdk import Choice, Noul, Score
from typesafe_sdk._core.response_types import SystemOneResponse
from system_one_models.decision_benchmark import (
    D1Backend, JevBackend, LayaBackend, create_backend, load_prompts, run_benchmark,
)


class BenchmarkTests(unittest.TestCase):
    def test_hosted_adapters_use_typed_sdk_and_explicit_credentials(self):
        scenario = load_prompts()[0]
        response = SystemOneResponse.model_validate({
            "model": "test", "usage": {"input_tokens": 42}, "answers": {
                "team": {"type": "choice", "choice": "billing", "confidence": .8,
                         "probabilities": {"billing": .8, "sales": .1, "support": .1}},
                "refund_ask": {"type": "noul", "noul": .1},
                "urgency": {"type": "score", "score": 2.7, "confidence": .8,
                            "legend": {0: "not urgent", 1: "normal", 2: "urgent", 3: "critical"},
                            "probabilities": {0: 0., 1: 0., 2: .3, 3: .7}},
            },
        })
        for cls in (JevBackend, D1Backend):
            with self.subTest(cls=cls), patch('typesafe_sdk.TypeSafeClient') as client:
                client.return_value.system_one.return_value = response
                backend = cls(api_key='explicit-test-key')
                result = run_benchmark([backend], [scenario])
                self.assertEqual(client.call_args.kwargs['api_key'], 'explicit-test-key')
                questions = client.return_value.system_one.call_args.kwargs['questions']
                self.assertIsInstance(questions['team'], Choice)
                self.assertIsInstance(questions['refund_ask'], Noul)
                self.assertIsInstance(questions['urgency'], Score)
                metric = result.metrics[0]
                self.assertEqual(metric['errors'], 0)
                self.assertEqual(metric['noul_acc'], 1)
                self.assertAlmostEqual(metric['score_mae'], .3)
                self.assertEqual(metric['avg_input_tokens'], 42)
                self.assertIsNone(metric['est_usd_per_1k'])
                backend.close()
                client.return_value.close.assert_called_once()

    def test_laya_unwraps_answers_and_passes_token(self):
        router = unittest.mock.Mock()
        router.return_value.predict.return_value = {'answers': {'q': {'noul': .9}}}
        with patch.dict('sys.modules', {'laya': SimpleNamespace(Router=router)}):
            backend = LayaBackend(token='test-token')
            self.assertEqual(backend.predict('state', {})['q']['noul'], .9)
            self.assertEqual(router.call_args.kwargs['token'], 'test-token')

    def test_mock_reports_are_reproducible_and_valid_json(self):
        def run():
            return run_benchmark([create_backend(n, mock=True) for n in ('jev', 'd1', 'laya')])
        first, second = run(), run()
        self.assertEqual(first.records, second.records)
        self.assertTrue(all(m['errors'] == 0 for m in first.metrics))
        with tempfile.TemporaryDirectory() as directory:
            first.write(directory)
            self.assertEqual(len(list(Path(directory).iterdir())), 3)
            data = json.loads((Path(directory) / 'results.json').read_text(),
                              parse_constant=lambda value: self.fail(value))
            self.assertEqual(data['records'], first.records)

    def test_failures_and_missing_answers_are_recorded(self):
        backend = create_backend('jev', mock=True)
        with patch.object(backend, 'predict', side_effect=RuntimeError('offline')):
            result = run_benchmark([backend], load_prompts()[:1])
            self.assertEqual(result.metrics[0]['errors'], 3)
            self.assertEqual(result.records[0]['error'], 'offline')
        with patch.object(backend, 'predict', return_value={}):
            result = run_benchmark([backend], load_prompts()[:1])
            self.assertEqual(result.metrics[0]['errors'], 3)
        with self.assertRaises(ValueError):
            create_backend('typo', mock=True)
