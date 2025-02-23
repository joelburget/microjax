# Microjax

<a target="_blank" href="https://colab.research.google.com/github/joelburget/microjax/blob/main/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> or <a href="https://github.com/joelburget/microjax/blob/main/tutorial.ipynb">Read on Github</a>.

This is inspired by Andrej Karpathy's [Micrograd](https://github.com/karpathy/micrograd/tree/master), a PyTorch-like library in about 150 lines of code. Despite PyTorch's popularity, I prefer the way Jax works. I think of PyTorch as having a more _object-oriented_ feel, while Micrograd has a more _functional_ feel. I think you'll start to see right away in the preview why that is.

This tutorial borrows heavily from Matthew J Johnson's great 2017 presentation on the predecessor to Jax, [autograd](https://github.com/hips/autograd): [Video](https://videolectures.net/videos/deeplearning2017_johnson_automatic_differentiation) / [Slides](https://www.cs.toronto.edu/~duvenaud/talks/Johnson-Automatic-Differentiation.pdf) / [Code](https://github.com/mattjj/autodidact). My main contribution is simplifying a bit and packaging it as a notebook.
