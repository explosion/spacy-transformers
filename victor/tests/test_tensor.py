import pytest
import numpy as np
from victor.tensor import OmegaTensor

def test_basic_ops_and_backward():
    """Tests simple addition, multiplication, and the backward pass."""
    a = OmegaTensor([2.0, 3.0], requires_grad=True, name="a")
    b = OmegaTensor([6.0, 4.0], requires_grad=True, name="b")

    # c = a + b
    # d = a * b
    # e = c + d
    e = (a + b) + (a * b)

    f = e.sum()

    a.zero_grad()
    b.zero_grad()

    f.backward()

    # d(f)/da = d(e)/da = d(a+b)/da + d(a*b)/da = 1 + b = [7, 5]
    expected_grad_a = np.array([7.0, 5.0])
    # d(f)/db = d(e)/db = d(a+b)/db + d(a*b)/db = 1 + a = [3, 4]
    expected_grad_b = np.array([3.0, 4.0])

    assert np.allclose(a.grad.data, expected_grad_a), "Gradient for 'a' is incorrect!"
    assert np.allclose(b.grad.data, expected_grad_b), "Gradient for 'b' is incorrect!"

def test_matmul_backward():
    """Tests matrix multiplication and its backward pass."""
    m1 = OmegaTensor(np.random.randn(2, 3), requires_grad=True, name="m1")
    m2 = OmegaTensor(np.random.randn(3, 4), requires_grad=True, name="m2")

    m3 = m1.matmul(m2)
    loss = m3.sum()

    m1.zero_grad()
    m2.zero_grad()

    loss.backward()

    assert m1.grad is not None, "m1 grad is None after matmul backward pass"
    assert m2.grad is not None, "m2 grad is None after matmul backward pass"
    assert m1.grad.shape == m1.shape, "m1 grad shape mismatch"
    assert m2.grad.shape == m2.shape, "m2 grad shape mismatch"

def test_activations():
    """Tests activation functions like tanh and sigmoid."""
    x = OmegaTensor(np.array([-1, 0, 1]), requires_grad=True, name="x")

    # Tanh
    t = x.tanh()
    t_sum = t.sum()
    x.zero_grad()
    t_sum.backward()
    expected_grad_t = 1 - np.tanh(x.data)**2
    assert np.allclose(x.grad.data, expected_grad_t), "Tanh grad failed."

    # Sigmoid
    s = x.sigmoid()
    s_sum = s.sum()
    x.zero_grad()
    s_sum.backward()
    sig_x = 1 / (1 + np.exp(-x.data))
    expected_grad_s = sig_x * (1 - sig_x)
    assert np.allclose(x.grad.data, expected_grad_s), "Sigmoid grad failed."

def test_reshape_and_transpose():
    """Tests reshape and transpose operations."""
    y = OmegaTensor(np.arange(6).reshape(2, 3), requires_grad=True, name="y")

    # Reshape
    y_reshaped = y.reshape((3, 2))
    y_reshaped_sum = y_reshaped.sum()
    y.zero_grad()
    y_reshaped_sum.backward()
    assert np.allclose(y.grad.data, np.ones((2, 3))), "Reshape grad failed."

    # Transpose
    y_transposed = y.transpose((1, 0))
    y_transposed_sum = y_transposed.sum()
    y.zero_grad()
    y_transposed_sum.backward()
    assert y_transposed.shape == (3, 2), "Transpose shape failed."
    assert np.allclose(y.grad.data, np.ones((2, 3))), "Transpose grad failed."

def test_cross_entropy_loss():
    """Tests the cross-entropy loss function."""
    logits = OmegaTensor(np.array([[2.0, 1.0, 0.1], [0.2, 0.5, 3.0]]), requires_grad=True, name="logits")
    targets = OmegaTensor(np.array([[1, 0, 0], [0, 0, 1]]), name="targets") # One-hot targets

    loss = logits.cross_entropy(targets)
    logits.zero_grad()
    loss.backward()

    # Check if gradient is reasonable (probs - target) / N
    probs = np.exp(logits.data) / np.exp(logits.data).sum(axis=1, keepdims=True)
    expected_grad_ce = (probs - targets.data) / logits.shape[0]

    assert np.allclose(logits.grad.data, expected_grad_ce), "Cross-Entropy grad failed."
