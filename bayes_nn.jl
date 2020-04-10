"""
Neural network constructor
"""
mutable struct network
    num_layers::Int64
    sizearr::Array{Int64,1}
    biases::Array{Array{Float64,1},1}
    weights::Array{Array{Float64,2},1}
    transfers::Array{Function,1}
    function network(sizes, activation_functions)
        num_layers = length(sizes)
        sizearr = sizes
        biases = [randn(y) for y in sizes[2:end]]
        weights = [randn(y, x) for (x, y) in zip(sizes[1:end-1], sizes[2:end])]
        transfers = [activation_functions[i] for i in 1:length(activation_functions)]
        return new(num_layers, sizearr, biases, weights, transfers)
    end
end

"""
Evaluate network on input x
"""
function nn_forward(net::network, x) # network always outputs layer is always linear
    for (w, b, h) in zip(net.weights, net.biases, net.transfers[2:end])
        x = h.(w*x .+ b)
    end
    return x
end

"""
Log posterior for binary classification (loglikelihood + logprior)
Loglikelihood can be obtained from the loss function implemented in Flux.
"""
function log_posterior(net::network, x, y, prior_sigma)
    return sum(Flux.binarycrossentropy.(nn_forward(net,x), y)) +
                0.5*norm(Flux.params([net.weights, net.biases]))^2 / prior_sigma^2
end

"""
Function that implements one iteration of the
Hamiltonian Monte Carlo algorithm to update the weights.
Leapfrog acts on matrices and arrays
"""
function hmc_bnn!(net::network, X, Y, prior_sigma, step_size, num_steps)

    p_for_biases = [randn(y) for y in net.sizearr[2:end]]
    p_for_weights = [randn(y, x) for (x, y) in zip(net.sizearr[1:end-1], net.sizearr[2:end])]
    current_K = 0.5 * norm([p_for_biases, p_for_weights])^2

    current_weights = copy(net.weights)
    current_biases = copy(net.biases)
    current_U = log_posterior(net, X, Y, prior_sigma)

    grads = Zygote.gradient(m -> log_posterior(m, X, Y, prior_sigma), net)[1][]
    # new state according to leapfrog for hamilton dynamics
    p_for_weights =  p_for_weights .- step_size .* grads.weights / 2.0
    p_for_biases = p_for_biases .- step_size .* grads.biases / 2.0
    for jump in 1:1:num_steps-1
        net.weights = net.weights .+ step_size .* p_for_weights# gradK.(p)
        net.biases = net.biases .+ step_size .* p_for_biases

        # update gradients to use with new  p
        grads = Zygote.gradient(m -> log_posterior(m, X, Y, prior_sigma), net)[1][]
        p_for_weights = p_for_weights .- step_size .* grads.weights
        p_for_biases = p_for_biases .- step_size .* grads.biases
    end
    net.weights = net.weights .+ step_size .* p_for_weights # gradK.(p)
    net.biases = net.biases .+ step_size .* p_for_biases

    # update gradients to use with new  p
    grads = Zygote.gradient(m -> log_posterior(m, X, Y, prior_sigma), net)[1][]

    # the second half for the kinetic variables
    p_for_weights = p_for_weights .- step_size .* grads.weights / 2.0
    p_for_biases = p_for_biases .- step_size .* grads.biases / 2.0   # the second half for the kinetic variables

    p_for_weights = -p_for_weights
    p_for_biases = -p_for_biases

    # test wethear we accept the new state or not
    # Evaluate potential and kinetic energies at start and end of trajectory
    proposed_U = log_posterior(net, X, Y, prior_sigma)
    proposed_K = 0.5 * norm([p_for_biases, p_for_weights])^2
    if  rand(Float64) .< exp(current_U - proposed_U + current_K - proposed_K)
        nothing
    else
        net.weights = current_weights
        net.biases = current_biases
    end
end

"""
Some helper functions.
"""
normalize_data(x) = (x .- mean(x)) ./ std(x);

function bnn_posterior!(net::network, weights, biases)
    net.weights = weights
    net.biases = biases
end

function cell2array(Z)
    m, n = size(Z)
    ZZ = zeros(m, n)
    for i in 1:1:m
        for j in 1:1:n
            ZZ[i,j] = Z[i,j][1]
        end
    end
    return ZZ
end
