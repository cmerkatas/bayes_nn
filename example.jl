"""
Import packages and files
"""
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

@info("Loading Packages...")
using Zygote, LinearAlgebra, Random
using Flux
using Random
using Plots

include("bayes_nn.jl")
"""
class_data.jl contains the data of the example
for Bayesian Neural Network with Turing language from Turinglang
    https://turing.ml/dev/tutorials/3-bayesnn/.
"""
include("turing_data.jl")    #
plot_data()

X = hcat(xs...)
Y = hcat(ts...)

"""
Set up a bnn for classification. The architecture and standard deviation are the
same as in the example of Turing in order to ensure correctness of the algorithm.
"""
# Create a regularization term and a Gaussain prior variance term.
alpha = 0.09
sig = sqrt(1.0 / alpha)

""" NNet with 2 input units, 3 units in 1st hidden layer,
2 units in 2nd hidden layer and 1 output unit and their correpsonding
activation function.
"""
ffnet = network([2, 3, 2, 1], [identity, tanh, tanh, sigmoid])

"""
Sample from the posterior distribution using Hamiltonian Monte Carlo.
"""
function main(net::network, X, Y, prior_sigma, maxiter, burnin)
    # matrices to store mcmc results
    sampled_weights = []
    sampled_bias = []
    lp = zeros(maxiter)

    for its in 1:maxiter
        hmc_bnn!(net, X, Y, prior_sigma, 0.05, 4)
        lp[its] = log_posterior(net, X, Y, prior_sigma)
        if its > burnin
            push!(sampled_weights, net.weights)
            push!(sampled_bias, net.biases)
        end
        if mod(its, 100) == 0
            println("HMC iterations: $its")
        end
    end

    return sampled_weights, sampled_bias, lp
end

Random.seed!(12345)
@time sampled_weights, sampled_bias, lp = main(ffnet, X, Y, sig, 5000, 1000)

# define test set
x_range = collect(range(-6,stop=6,length=25))
y_range = collect(range(-6,stop=6,length=25))

"""
Visualise the results
"""
n_end = 4000
anim = @gif for i=1:100:n_end
    plot_data();
    bnn_posterior!(ffnet, sampled_weights[i], sampled_bias[i])
    Z = [nn_forward(ffnet, [x, y]) for x=x_range, y=y_range]
    contour!(x_range, y_range, cell2array(Z), title="Iteration $i", clim = (0,1));
end every 10;


_, idx = findmax(lp)
bnn_posterior!(ffnet, sampled_weights[idx], sampled_bias[idx])
Z = [nn_forward(ffnet, [x, y]) for x=x_range, y=y_range]
plot_data()
contour!(x_range, y_range, cell2array(Z))
