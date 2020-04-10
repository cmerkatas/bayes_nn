# bayes_nn
Bayesian Neural Networks for binary classifcation.
The Hamiltonian Monte Carlo algorithm is used for posterior inference.

The data for classification example are taken from the tutorial on Bayesian neural networks of the [Turing](https://turing.ml/) Probabilistic Language.

## To do
1. Modify the log-posterior to perform regression tasks.
2. Re-implement the sampler using the interface of an ```AbstractSampler``` from [```AbstractMCMC```](https://github.com/cmerkatas/AbstractMCMC.jl) package.
3. Add more MCMC algorithms. In particular, Stochastic Gradient variants of the HMC and generalizations.


