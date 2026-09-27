"""Unit tests: shapes, recurrence equations against a NumPy reference, and the
ability of every cell to fit a tiny sine series.  Run: python -m unittest -v tests.test_cells"""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cells import SimpleRNN  # noqa: E402

torch.set_num_threads(1)


def np_params(m):
    return {k: v.detach().numpy().astype(float) for k, v in m.named_parameters()}


class TestCells(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(3, 7, 1)

    def test_shapes(self):
        for kind in ["elman", "jordan", "mrn"]:
            m = SimpleRNN(kind, hidden_size=5)
            self.assertEqual(tuple(m(self.x).shape), (3, 7))

    def test_elman_equations(self):
        m = SimpleRNN("elman", 4)
        P = np_params(m)
        x = self.x.numpy().astype(float)
        h = np.zeros((3, 4))
        ys = []
        for t in range(7):
            h = np.tanh(x[:, t] @ P["cell.W_xh.weight"].T + P["cell.W_xh.bias"] + h @ P["cell.W_ch.weight"].T)
            ys.append(h @ P["out.weight"].T + P["out.bias"])
        np.testing.assert_allclose(m(self.x).detach().numpy(), np.hstack(ys), atol=1e-5)

    def test_jordan_equations(self):
        m = SimpleRNN("jordan", 4)
        P = np_params(m)
        x = self.x.numpy().astype(float)
        s = np.zeros((3, 1))
        ys = []
        for t in range(7):
            h = np.tanh(x[:, t] @ P["cell.W_xh.weight"].T + P["cell.W_xh.bias"] + s @ P["cell.W_sh.weight"].T)
            y = h @ P["out.weight"].T + P["out.bias"]
            ys.append(y)
            s = y                                   # state = previous output
        np.testing.assert_allclose(m(self.x).detach().numpy(), np.hstack(ys), atol=1e-5)

    def test_mrn_equations(self):
        a = np.array([0.25, 0.5, 0.75])
        m = SimpleRNN("mrn", 4, decays=tuple(a))
        P = np_params(m)
        x = self.x.numpy().astype(float)
        Mh, My = np.zeros((3, 3, 4)), np.zeros((3, 3, 1))
        h, y = np.zeros((3, 4)), np.zeros((3, 1))
        ys = []
        for t in range(7):
            Mh = a[None, :, None] * Mh + (1 - a[None, :, None]) * h[:, None, :]
            My = a[None, :, None] * My + (1 - a[None, :, None]) * y[:, None, :]
            mem = np.concatenate([Mh.reshape(3, -1), My.reshape(3, -1)], 1)
            h = np.tanh(x[:, t] @ P["cell.W_xh.weight"].T + P["cell.W_xh.bias"] + mem @ P["cell.W_mh.weight"].T)
            y = h @ P["out.weight"].T + P["out.bias"]
            ys.append(y)
        np.testing.assert_allclose(m(self.x).detach().numpy(), np.hstack(ys), atol=1e-5)

    def test_mrn_zero_decay_is_elman_plus_jordan_feedback(self):
        """With a single bank and a=0 the MRN memory is exactly [h_{t-1}; y_{t-1}]."""
        m = SimpleRNN("mrn", 3, decays=(0.0,))
        st = m.cell.init_state(2)
        h_prev, y_prev = torch.randn(2, 3), torch.randn(2, 1)
        st = (st[0], st[1], h_prev, y_prev)
        _, (Mh, My, _, _) = m.cell(torch.randn(2, 1), st, m.out)
        torch.testing.assert_close(Mh[:, 0], h_prev)
        torch.testing.assert_close(My[:, 0], y_prev)

    def test_fits_tiny_sine(self):
        t = np.arange(120)
        s = np.sin(2 * np.pi * t / 12).astype(np.float32)
        W = 12
        idx = np.arange(W, len(s))
        X = torch.from_numpy(np.stack([s[i - W:i] for i in idx])[..., None])
        Y = torch.from_numpy(np.stack([s[i - W + 1:i + 1] for i in idx]))
        for kind in ["elman", "jordan", "mrn"]:
            torch.manual_seed(0)
            m = SimpleRNN(kind, 8)
            opt = torch.optim.Adam(m.parameters(), lr=1e-2)
            first = None
            for _ in range(300):
                opt.zero_grad()
                loss = torch.mean((m(X)[:, -1] - Y[:, -1]) ** 2)
                first = loss.item() if first is None else first
                loss.backward()
                opt.step()
            self.assertLess(loss.item(), 0.1 * first, f"{kind} failed to fit sine")  # >90% loss reduction


if __name__ == "__main__":
    unittest.main(verbosity=2)
