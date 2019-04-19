using Flux, Distributions, Random
using CSV, DataFrames, Plots
using StatsPlots, KernelDensity
using LinearAlgebra

include("bayes_nn.jl")
# An example from Turing.jl

# Number of points to generate.
N = 80
M = round(Int, N / 4)
Random.seed!(1234)

# Generate artificial data.
x1s = rand(M) * 4.5; x2s = rand(M) * 4.5;
xt1s = Array([[x1s[i] + 0.5; x2s[i] + 0.5] for i = 1:M])
x1s = rand(M) * 4.5; x2s = rand(M) * 4.5;
append!(xt1s, Array([[x1s[i] - 5; x2s[i] - 5] for i = 1:M]))

x1s = rand(M) * 4.5; x2s = rand(M) * 4.5;
xt0s = Array([[x1s[i] + 0.5; x2s[i] - 5] for i = 1:M])
x1s = rand(M) * 4.5; x2s = rand(M) * 4.5;
append!(xt0s, Array([[x1s[i] - 5; x2s[i] + 0.5] for i = 1:M]))

# Store all the data for later.
xs = [xt1s; xt0s]
ts = [ones(2*M); zeros(2*M)]

xs = hcat(xs...)'
# Plot data points.
function plot_data()
    x1 = map(e -> e[1], xt1s)
    y1 = map(e -> e[2], xt1s)
    x2 = map(e -> e[1], xt0s)
    y2 = map(e -> e[2], xt0s)

    Plots.scatter(x1,y1, color="red", clim = (0,1))
    Plots.scatter!(x2, y2, color="blue", clim = (0,1))
end

plot_data()

network_shape = [
    (3,2, :tanh),
    (2,3, :tanh),
    (1,2, :σ)]

alpha = 0.09
sig = sqrt(1.0 / alpha)

num_params = sum([i * o + i for (i, o, _) in network_shape]);
nsamples=5000

samples = rand(Normal(0, sig), nsamples, num_params)


lp=zeros(nsamples)
precisions = ones(nsamples, size(network_shape)[1])
hypers = [0.001 0.001;
          0.001 0.001;
          0.001 0.001]
hyper_tau = []
Random.seed!(12345)

@time for i in 2:nsamples
    samples[i,:] = HMC(xs, ts, samples[i-1,:], network_shape, precisions[i - 1,:], [] , 0.05, 4)
    precisions[i,:] = update_precisions(xs, ts, samples[i,:], network_shape, hypers, hyper_tau)
    lp[i] = log_posterior(xs, ts, samples[i-1,:], network_shape, precisions[i-1,:], [])
    if mod(i,1000).==0
        print("MCMC iterations: $i\n")
    end
end

plot_data()

x_range = collect(range(-6,stop=6,length=25))
y_range = collect(range(-6,stop=6,length=25))
Z = [nn_predict([x, y]', samples, nsamples, network_shape)[1] for x=x_range, y=y_range]
contour!(x_range, y_range, Z)
