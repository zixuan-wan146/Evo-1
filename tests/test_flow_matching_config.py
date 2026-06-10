import unittest


class FlowMatchingConfigTests(unittest.TestCase):
    def test_action_head_can_be_constructed_without_config(self):
        torch = self._import_or_skip("torch")
        flow_matching = self._import_or_skip("Evo_1.model.action_head.flow_matching")

        head = flow_matching.FlowmatchingActionHead(
            embed_dim=8,
            hidden_dim=16,
            action_dim=6,
            horizon=2,
            per_action_dim=3,
            num_heads=2,
            num_layers=1,
        )

        self.assertIsInstance(head, torch.nn.Module)
        self.assertEqual(head.horizon, 2)
        self.assertEqual(head.per_action_dim, 3)
        self.assertEqual(head.action_dim, 6)

    def _import_or_skip(self, module_name):
        try:
            return __import__(module_name, fromlist=["*"])
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional dependency unavailable for this test: {exc.name}")


if __name__ == "__main__":
    unittest.main()
