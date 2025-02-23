import unittest
import numpy
from microjax import grad


def relu(x):
    return (x + abs(x)) / 2


def neuron(w, b, x, nonlin=True):
    z = w @ x + b
    return relu(z) if nonlin else z


layer = neuron


def mlp(layers, x):
    for w, b in layers[:-1]:
        x = layer(w, b, x)
    w, b = layers[-1]
    return layer(w, b, x, nonlin=False)


def assert_allclose(a, b):
    numpy.testing.assert_allclose(a, b, rtol=1e-5, atol=0)


class TestRelu(unittest.TestCase):
    def test_relu(self):
        self.assertAlmostEqual(relu(1.0), 1.0)
        self.assertAlmostEqual(relu(-1.0), 0.0)
        self.assertAlmostEqual(grad(relu)(1.0), 1.0)
        self.assertAlmostEqual(grad(relu)(-1.0), 0.0)
        self.assertAlmostEqual(grad(grad(relu))(1.0), 0.0)
        self.assertAlmostEqual(grad(grad(relu))(-1.0), 0.0)


class TestNeuron(unittest.TestCase):
    def test_call(self):
        w = numpy.array([[1.0, 2.0]])
        b = 0.0
        x = numpy.array([1.0, 2.0])
        numpy.testing.assert_allclose(neuron(w, b, x), numpy.array([5]))

    def test_value_and_grad(self):
        w = numpy.ones((2,))
        b = 1.0
        x = numpy.ones((2,))
        assert_allclose(neuron(w, b, x), 3.0)
        assert_allclose(grad(lambda w: neuron(w, b, x))(w), numpy.array([1.0, 1.0]))
        assert_allclose(grad(lambda b: neuron(w, b, x))(b), numpy.array([1.0]))


class TestLayer(unittest.TestCase):
    def test_value_and_grad(self):
        w = numpy.array([1.0, 2.0])
        b = 0.0
        x = numpy.ones((2,))
        assert_allclose(layer(w, b, x), 3.0)
        assert_allclose(grad(lambda w: layer(w, b, x))(w), numpy.array([1.0, 1.0]))
        assert_allclose(grad(lambda b: layer(w, b, x))(b), 1.0)

        w = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        b = numpy.zeros((2,))

        assert_allclose(
            grad(lambda w: w @ numpy.array([1.0, -1.0]))(numpy.array([3.0, 7.0])),
            numpy.array([1.0, -1.0]),
        )

        assert_allclose(
            grad(lambda w: (w @ x) @ numpy.array([1.0, -1.0]))(w),
            numpy.array([[1.0, 1.0], [-1.0, -1.0]]),
        )

        assert_allclose(
            grad(lambda w: layer(w, b, x) @ numpy.array([1.0, -1.0]))(w),
            numpy.array([[1.0, 1.0], [-1.0, -1.0]]),
        )


class TestMLP(unittest.TestCase):
    def test_value_and_grad(self):
        w1 = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        w2 = numpy.array([[5.0, 6.0], [7.0, 8.0]])
        b1 = numpy.zeros((2,))
        b2 = b1
        x = numpy.array([1.0, 2.0])
        assert_allclose(mlp([(w1, b1), (w2, b2)], x), numpy.array([91.0, 123.0]))
        assert_allclose(
            grad(lambda w1: mlp([(w1, b1)], x) @ numpy.array([1.0, -1.0]))(w1),
            numpy.array([[1.0, 2.0], [-1.0, -2.0]]),
        )

        assert_allclose(
            grad(lambda w2: mlp([(w1, b1), (w2, b2)], x) @ numpy.array([1.0, -1.0]))(
                w2
            ),
            numpy.array([[5.0, 11.0], [-5.0, -11.0]]),
        )

        assert_allclose(
            grad(lambda b1: mlp([(w1, b1), (w2, b2)], x) @ numpy.array([1.0, -1.0]))(
                b1
            ),
            numpy.array([-2.0, -2.0]),
        )

        assert_allclose(
            grad(lambda b2: mlp([(w1, b1), (w2, b2)], x) @ numpy.array([1.0, -1.0]))(
                b2
            ),
            numpy.array([1.0, -1.0]),
        )


if __name__ == "__main__":
    unittest.main()
