# GeoCert

Geometric-inspired algorithm for solving the general problem of finding the largest l_p ball centered at a point x for a union of polytopes which form a polyhedral complex. This algorithm is provably correct for p equal or larger than 1. Primary application found in certifying the adversarial robustness of multilayer ReLu networks. Created by [Matt Jordan](https://cs.utexas.edu/~mjordan) and Justin Lewis. 

Check out our paper on arXiv: [Provable Certificates for Adversarial Examples: Fitting a Ball in the Union of Polytopes](https://arxiv.org/abs/1903.08778).


Some example results:

Maximal l_2 projections           | Network Input Partioning
-----------------------------------------|-----------------------------------------
<img src="https://github.com/revbucket/geometric-certificates/blob/master/assets/non_robust_projections_l2.svg" alt="example" width="400"> | <img src="https://github.com/revbucket/geometric-certificates/blob/master/assets/locus_PG_plot_non_robust.svg" alt="mnist_reconstr" width="400">

---
# News
- 09/13/2019: Version 0.2 refactor deployed
- 09/03/2019: Accepted (poster) to NeurIPS 2019
- 06/11/2019: Contributed Talk at ICML Workshop on [Security and Privacy of Machine Learning](https://icml2019workshop.github.io)
- 03/20/2019: ArXiv Release and Version 0.1 deployed
# Primary Contents


### Functions:
- Computing the minimal distance adversarial example under L<sub>2</sub> and L<sub>&#8734;</sub> norms. That is, given a classifier _f_ and an input _x_, GeoCert computes the &delta; with minimal L<sub>p</sub> norm such that f(x) &ne; f(x+&delta;).
- Answering the decision problem of robustness. Given a classifier _f_, an input _x_ and a radius &epsilon;, answer the decision problem: "Does there exist a point _y_ with ||y-x||<sub>p</sub> &le; &epsilon; such that f(y)&ne; f(x)?"
- Recalling that ReLU neural networks are piecewise linear, GeoCert can be leveraged to exactly count the number of linear regions intersecting a specified L<sub>p</sub> ball.

### Examples:
* __2D_example.ipynb:__ Basic example using a binary classifier on 2-dimensional inputs to demonstrate the primary functionalities of GeoCert
* __MNIST_example.ipynb:__ More fleshed-out example, where the classifier is trained to distinguish between 1's and 7's from the MNIST Dataset.

---

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Dependencies
Requisite python packages are contained within the file `requirements.txt`. The [`mister_ed`](https://github.com/revbucket/mister_ed) adversarial example toolbox is used to compute upper bounds. This is maintained as a subrepository within this one.

GeoCert makes many many calls to linear program solvers (in the $\ell_\infty$ case) or LCQP solvers (in the $\ell_2$) case. We use the [Gurobi Optimizer](https://www.gurobi.com) for this. Visit their homepage to acquire a free academic license.

### Installing

1. Install the repository:
    ```shell
    $ pip install git+https://github.com/revbucket/geometric-certificates.git
    ```
---

## Running the tests

With the codebase installed, run the ipython notebooks provided to get your hands on the algorithm and visualize its behaviour. As an example:

```shell
$ cd examples 
$ jupyter notebook 2D_example.ipynb
```	

---

## Authors

* **Matt Jordan-** University of Texas at Austin 
* **Justin Lewis-** University of Texas at Austin 


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


