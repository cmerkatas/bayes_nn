using Flux, Distributions, Random
using CSV, DataFrames, Plots
using Turing, StatsPlots, KernelDensity
### data normalizer
normalize_data(x) = (x .- mean(x)) ./ std(x);

# This modification of the unpack function generates a series of vectors
# given a network shape.
# Originally taken from turing.ml
function unpack(θ::AbstractVector, network_shape::AbstractVector)
  index = 1
  weights = []
  biases = []
  for layer in network_shape
      rows, cols, _ = layer
      size = rows * cols
      last_index_w = size + index - 1
      last_index_b = last_index_w + rows
      push!(weights, reshape(θ[index:last_index_w], rows, cols))
      push!(biases, reshape(θ[last_index_w+1:last_index_b], rows))
      index = last_index_b + 1
  end
  return weights, biases
end;

# Generate an abstract neural network given a shape,
# and return a prediction.
# Originally taken from turing.ml
function nn_forward(x, θ::AbstractVector, network_shape::AbstractVector)
  weights, biases = unpack(θ, network_shape)
  layers = []
  for i in eachindex(network_shape)
      push!(layers, Dense(weights[i],
          biases[i],
          eval(network_shape[i][3])))
  end
  nn = Chain(layers...)
  return nn(x)
end;


# define log posterior density for classification or regression
function log_posterior(x, y, θ::AbstractVector, network_shape::AbstractVector, precisions::Array{Float64}, τ)
  if isempty(τ)
    log_like = -sum(Flux.binarycrossentropy.(nn_forward(x',θ,network_shape), y'))
    weights, biases = unpack(θ, network_shape)
    for layer in 1:1:size(weights)[1]
      θ = vcat(sqrt(precisions[layer]) .* weights[layer][:], sqrt(precisions[layer]) .* biases[layer][:])
    end
    return -(log_like - 0.5norm(θ,2)^2)
  else
    n = length(y)
    log_like = 0.5n * log(τ) - 0.5n * log(2π) - 0.5τ * Flux.mse(y, nn_forward(x', θ, network_shape))
    weights, biases = unpack(θ, network_shape)
    for layer in 1:1:size(weights)[1]
      θ = vcat(sqrt(precisions[layer]) .* weights[layer][:], sqrt(precisions[layer]) .* biases[layer][:])
    end
    return -(log_like - 0.5norm(θ,2)^2)
  end
end;

# gradients of the log posterior density with respect to parameters
function ∇log_posterior(x, y, θ::AbstractVector, network_shape::AbstractVector, precisions::Array{Float64}, τ)
  temp = Tracker.gradient(() -> log_posterior(x, y, θ, network_shape, precisions, τ), Flux.params(θ))
  return collect(temp[θ])
end

# function for a single iteration of the Hamiltonian monte carlo algorithm
function HMC(x, y, current_q, network_shape::AbstractVector, precisions::Array{Float64}, τ, ϵ = .1, L = 10)
  q = current_q
  p = rand(Normal(0, 1), length(q))  # independent standard normal variates
  current_p = p

  ## prepare gradients
  q = param(q)

  # Make a half step for momentum at the beginning
  p = p .- ϵ .* ∇log_posterior(x, y, q, network_shape, precisions, τ) ./ 2
  # Alternate full steps for position and momentum
  for i in 1:1:L
    # Make a full step for the position
    q = q .+ ϵ .* p
    # Make a full step for the momentum, except at end of trajectory
    if i != L
      p = p .- ϵ .* ∇log_posterior(x, y, q, network_shape, precisions, τ)
    end
  end

  # Make a half step for momentum at the end.

  p = p .- ϵ * ∇log_posterior(x, y, q, network_shape, precisions, τ) ./ 2
  # Negate momentum at end of trajectory to make the proposal symmetric
  p = -p
  # Evaluate potential and kinetic energies at start and end of trajectory
  current_U = log_posterior(x, y, current_q, network_shape, precisions, τ)
  current_K = norm(current_p,2)^2 ./ 2
  proposed_U = log_posterior(x, y, q, network_shape, precisions, τ)
  proposed_K = norm(p,2)^2 ./ 2
  # Accept or reject the state at end of trajectory, returning either
  # the position at the end of the trajectory or the initial position
  if log(rand()) < current_U - proposed_U + current_K - proposed_K
    # acc <<- acc + 1
    current_q = q # accept
  end
  return collect(current_q)
end

function update_precisions(x, y, θ::AbstractVector, network_shape::AbstractVector, hyperparams::Array{Float64,2}, hyper_tau=[])
  weights, biases = unpack(θ, network_shape)
  nlayers = size(weights)[1]
  precisions = zeros(nlayers)
  for layer in 1:1:nlayers
    layer_weights = vcat(weights[layer][:], biases[layer][:])
    alpha_star = hyperparams[layer, 1] + 0.5length(layer_weights)
    beta_star = hyperparams[layer, 2] + 0.5norm(layer_weights,2)^2
    precisions[layer] = rand(Gamma(alpha_star, 1/beta_star))
  end
  if isempty(hyper_tau)
    return precisions
  else
    alpha_τ = hyper_tau[1] + 0.5size(x)[1]
    beta_τ = hyper_tau[2] #+ 0.5Flux.mse(y, nn_forward(x', θ, network_shape))
    tau = rand(Gamma(alpha_τ, 1/beta_τ))
    return precisions, tau
  end
end


# Reorder the sampled weights
function reorder!(weights::Array{Float64,2})
  new_order=sortperm(vec(mean(weights,dims=1)))
  weights[:,new_order]
end

# This function makes predictions based on network shape.
# Return the average predicted value across
# multiple weights.
# Originally taken from turing.ml
function nn_predict(x, theta, num, network_shape)
    mean([nn_forward(x', theta[i,:], network_shape) for i in 1:1000:num])
end;
