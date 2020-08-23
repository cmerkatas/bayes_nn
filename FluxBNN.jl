using Flux, Distributions, Random
using Flux: onehotbatch, Zygote
using AdvancedHMC
using LinearAlgebra
using Plots
include("turing_data.jl")
include("utils.jl")

struct BayesNNet{N, P, RE}
    nnet::N
    p::P
    re::RE

    function BayesNNet(nnet;p = nothing)
        _p, re = Flux.destructure(nnet)
        if p === nothing
            p = _p
        end
        new{typeof(nnet), typeof(p), typeof(re)}(
            nnet, p, re)
    end
end

(b::BayesNNet)(x, p=b.p) = b.re(p)(x)

net_topology = Chain(Dense(2,3,tanh), Dense(3,2,tanh), Dense(2,2,σ))
bnn = BayesNNet(net_topology)
# bnn(x)

x_train = Flux.normalise(X,dims=2)
y_train = onehotbatch(Y, 0:1)

# log posterior density
alpha = 0.09
prior_sigma = sqrt(1/alpha)
function ℒ(p)
    -sum(Flux.binarycrossentropy.(bnn(x_train, p), y_train)) -
        0.5*norm(p)^2 / prior_sigma^2
end


# Set the number of samples to draw and warmup iterations
n_samples, n_adapts = 6_000, 2_000

# Define a Hamiltonian system
D = length(bnn.p)
metric = DiagEuclideanMetric(D)
hamiltonian = Hamiltonian(metric, ℒ, Zygote)

Random.seed!(1)
initial_p = randn(D)

# Define a leapfrog solver, with initial step size chosen heuristically
initial_ϵ = find_good_stepsize(hamiltonian, initial_p)
integrator = Leapfrog(initial_ϵ)

# Define an HMC sampler, with the following components
#   - multinomial sampling scheme,
#   - generalised No-U-Turn criteria, and
#   - windowed adaption for step-size and diagonal mass matrix
proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.4, integrator))

# Run the sampler to draw samples from the specified Gaussian, where
#   - `samples` will store the samples
#   - `stats` will store diagnostic statistics for each sample
Random.seed!(123)
samples, stats = sample(hamiltonian, proposal, initial_p, n_samples, adaptor, n_adapts;
                        drop_warmup=true, progress=true)

# define test set
x_range = collect(range(-6,stop=6,length=25))
y_range = collect(range(-6,stop=6,length=25))
n_end = size(sampled_vals, 2)
anim = @gif for i=1:20:n_end
    plot_data();
    Z = [bnn([x, y], samples[i]) for x=x_range, y=y_range]
    contour!(x_range, y_range, cell2array(Z), title="Iteration $i", clim = (0,1));
end every 30;
