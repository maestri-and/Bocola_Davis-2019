###############################################################################
############################### PARAMETERS.JL #################################

############ This script defines and assigns the parameters of ################
################ the benchmark Bocola & Dovis (2019) model ####################

###############################################################################


###############################################################################
#                               Import libraries
###############################################################################

using LinearAlgebra
# using IterTools - TO BE DELETED
using Base.Iterators

include("auxiliary_functions.jl")

###############################################################################
#                        Define structural parameters
###############################################################################

# Government's decision problem parameters

sigma   = 2.000                 # Risk aversion
psi     = 0.050                 # Probability of re-entry in capital markets
tau     = 0.410                 # Tax revenues over GDP
g_star       = 0.680                 # Nondiscretionary spending over tax revenues
mu_y    = 0.892                 # Normalization
rho_y   = 0.970                 # Persistence of output
sigma_y = 0.008                 # Volatility of output innovations
sigma_yk = -0.002               # Loading of chi innovations on output (for bounds)
sigma_yk2 = -0.0024             # Loading of chi innovations on output (for parameters)
beta    = 0.980                 # Government discount factor
#d0      = 0.058                 # Output losses in low income states
#d1      = 0.092                 # Average output losses
d0     = 0.140625;             # Output Losses in low income states
d1     = 0.225;                # Average Output Losses
alp   = alpha = 0.400                 # Adjustment cost, maturity, elasticity
# TO DO -Ensure consistency of alp/alpha naming
d_upper    = 6.810                 # Adjustment cost, maturity, level
pi_star  = -6.500                # Mean of pi
sigma_pi = 1.650                # Volatility of pi

# Stochastic discount factor parameters
phi0    = 0.0049                 # Constant term in sdf
phi1    = 0.0016                 # Loading on chi, sdf
kappa0  = 0.1605                 # Constant, market price of risk
kappa1  = 0.3738                 # Loading on chi, market price of risk
rho_chi = 0.5126                 # Persistence of chi
mu_chi  = 0                     # Constant in chi_t process
sigma_chi = 1                   # Volatility of chi

# Comments on discrepancies:
# - phi0: 0.005 (paper) vs. 0.0049 (model_solution.m)
# - phi1: 0.002 (paper) vs. 0.0016 (model_solution.m)
# - kappa0: 0.161 (paper) vs. 0.1605 (model_solution.m)
# - kappa1: 0.374 (paper) vs. 0.3738 (model_solution.m)
# - d0: 0.058 (paper) vs. 0.140625 (model_solution.m)
# - d1: 0.092 (paper) vs. 0.225 (model_solution.m)
# - pi_bar: -6.500 (paper) vs. -6.5 (model_solution.m)

###############################################################################
#                        Define computational parameters
###############################################################################

state_ex    = 3;              # Number of exogenous state variables
N           = 5;              # Number of points for each exogenous state in value function
N_ex        = N^state_ex;          # Numper of grid points for value function approximation
N_b         = 81;             # Number of grid points for b in value function
b_min       = 0;              # Lower bound for grid of b
b_max       = 16;             # Uper bound for grid of b

N_l         = 11;             # Number of grid points for lambda in value function
lam_min     = 5;              # Lower bound for grid of lambda
lam_max     = 8;              # Uper bound for grid of lambda

N_s         = 650;          # Number of points for bprime
N_b1        = 50;           # Number of points in first subgrid of bprime
N_b3        = 100;          # Number of points in last subgrid of bprime
b1_min      = 0;            # Lower bound for first subgrid of bprime
b1_max      = 5.99;         # Uper bound for first subgrid of bprime
b2_min      = 6;            # Lower bound for middle subgrid of bprime
b2_max      = 11.99;        # Uper bound for middle subgrid of bprime
b3_min      = 12;           # Lower bound for last subgrid of bprime
b3_max      = b_max;        # Uper bound for last subgrid of bprime
N_bb        = 500;          # Number of points arround b to choose bprime

N_py        = 51;           # Number of points for y in pricing schedule
N_pchi      = 5;            # Number of points for chi in pricing schedule
N_ppi       = 5;            # Number of points for pi in pricing schedule
N_price     = N_py*N_pchi*N_ppi;# Number of grid points for pricing schedule 
N_ps        = N_py*N_pchi*N_ppi; # TO DO: ensure consistency of N_price/N_ps naming 

N_p         = 3^state_ex;   # Number of points in Gauss-Hermite quadrature
N_pq        = 5^state_ex;   # Number of points in second Gauss-Hermite quadrature for price

N_ghq       = 3;            # Number of points in Gauss-Hermite quadrature
N_ghq1      = 5;            # Number of points in second Gauss-Hermite quadrature for price

max_iter    = 550;          # Maximum number of iterations

convergence_q = 0.999;      # Smoothing of pricing schedule
convergence_v = 0.00;       # Smoothing of value function
val_lb        = 0.02;       # Lower bound for value function
lwb           = 0.75;       # Lower bound for debt grid
uwb           = 1.25;       # Upper bound for debt grid
start_conv    = 1;    
tolerance     = 10^(-4);

###############################################################################
#                        Allocate further variables 
###############################################################################

# Integer variables
# i = 0
# j = 0
# k = 0
# l = 0
# m = 0
# n = 0
# kk = 0
# o = 0
# p = 0
# index_b = zeros(Int, 1)
# maxb = zeros(Int, 1)
# maxl = zeros(Int, 1)
# index_ss = zeros(Int, 2)
# maxg = zeros(Int, 1)

# Model Variables
price_index = zeros(Int, 1, N_ex)
# ss = zeros(Float32, state_ex, 1)
# bounds = zeros(Float32, state_ex, 2)
# gamma_a = zeros(Float32, 1, N_ex)
# d = zeros(Float32, N_ex)
# gamma_prov = zeros(Float32, 1, N_ex)
# gamma_p = zeros(Float32, N_l, N_b, N_ex)
# gamma_ck = zeros(Float32, N_l, N_b, N_ex)
# gamma_pnew = zeros(Float32, N_l, N_b, N_ex)
# gamma_cknew = zeros(Float32, N_l, N_b, N_ex)
# lamprime = zeros(Int, N_l, N_b, N_ex)
# lamprime_old = zeros(Int, N_l, N_b, N_ex)
# bprime = zeros(Int, N_l, N_b, N_ex)
# bprime_old = zeros(Int, N_l, N_b, N_ex)
# debt_choice_old = zeros(Float32, N_l, N_b, N_ex)
# maturity_choice_old = zeros(Float32, N_l, N_b, N_ex)
# value_paynew = zeros(Float32, N_l, N_ex, N_b)
# value_payold = zeros(Float32, N_l, N_ex, N_b)
# value_cknew = zeros(Float32, N_l, N_ex, N_b)
# value_ckold = zeros(Float32, N_l, N_ex, N_b)
# qeq_old = zeros(Float32, N_l, N_ex, N_b)
# qeq_new = zeros(Float32, N_l, N_ex, N_b)
# TT = zeros(Float32, N_ex, N_ex)
# InvTT = zeros(Float32, N_ex, N_ex)
# Bline = zeros(Float32, N_s, 1)
# debt = zeros(Float32, N_b, 1)
# lam = zeros(Float32, N_l, 1)
# coll_points = zeros(Float32, state_ex, N_ex)
coll_price = zeros(Float32, state_ex, N_price)
# weights = zeros(Float32, 1, N_p)
weightsq = zeros(Float32, 1, N_pq)
points = zeros(Float32, state_ex, N_p)
pointsq = zeros(Float32, state_ex, N_pq)
q_new = zeros(Float32, N_s, N_l, N_l, N_price)
q_old = zeros(Float32, N_s, N_l, N_l, N_price)

# Aux variables for process
y_y = zeros(Float32, N_ex)
chi_y = zeros(Float32, N_ex)
pi_y = zeros(Float32, N_ex)
# x_prime = zeros(Float32, state_ex, 1)
# y_prime = zeros(Float32, state_ex, 1)
extra = zeros(Float32, state_ex, 1)
extra2 = zeros(Float32, state_ex, 1)
# y_prime_TP = zeros(Float32, 1, state_ex)
y_p = zeros(Float32, 1, N_price)
pi_p = zeros(Float32, 1, N_price)
chi_p = zeros(Float32, 1, N_price)
TTT1 = zeros(Float32, N_ex, N_p, N_ex, 1)
TTT2 = zeros(Float32, N_price, N_pq, N_ex, 1)

# Aux variables for setup
points_y = zeros(Float32, N, 1)
points_chi = zeros(Float32, N, 1)
points_pi = zeros(Float32, N, 1)
y_temp = zeros(Float32, N_ex, state_ex)
points_qy = zeros(Float32, N_py, 1)
points_qchi = zeros(Float32, N_pchi, 1)
points_qpi = zeros(Float32, N_ppi, 1)
B1 = zeros(Float32, N_b1, 1)
B2 = zeros(Float32, N_s - N_b1 - N_b3, 1)
B3 = zeros(Float32, N_b3, 1)
T = zeros(Float32, N_ex, N, state_ex)
lam_back = zeros(Float32, N_l, 1)
Tprod = zeros(Float32, N_ex, N_ex)

# Indexes
index_s = zeros(Int, N_s, 1)
N_sl = zeros(Int, N_b, 1)
N_su = zeros(Int, N_b, 1)
index_ck = zeros(Int, N_b, N_l, 1)
index_ex2 = zeros(Int, 3, N_price, N_pq)
index_ex1 = zeros(Int, 3, N_price, N_pq)
s_prime = zeros(Float32, 3, N_price, N_pq)


# #Model variables
# value_aut = zeros(Float32, N_ex, 1)
# gam_bound = zeros(Float32, 1, N_ex)

# value_pay = zeros(Float32, N_ex, N_l, N_b, N_bb, N_l)

# value_ck = zeros(Float32, N_l, N_b, N_ex)
# maxv = zeros(Float32, N_l, N_b, N_ex)
# maxv1 = zeros(Float32, N_ex, N_l, N_b, N_bb)
# lamprime1 = zeros(Float32, N_l, N_b, N_ex, N_bb)

###############################################################################
#                        Bounds for exogenous state variables 
###############################################################################

# Steady state values
ss = [0; mu_chi / (1 - rho_chi); exp(pi_star) / (1 + exp(pi_star))]

#Bounds
bounds_y    = [-3*((sigma_y-sigma_yk)^(2)/(1-0.97^(2)))^(1/2),
                3*((sigma_y-sigma_yk)^(2)/(1-0.97^(2)))^(1/2)];
bounds_chi  = [-3*((sigma_chi)^(2)/(1-rho_chi^(2)))^(1/2),
                3*((sigma_chi)^(2)/(1-rho_chi^(2)))^(1/2)];
bounds_pi   = [0,
                exp(pi_star+3.5*sigma_pi)./(1+exp(pi_star+3.5*sigma_pi))];

bounds = vcat(bounds_y', bounds_chi', bounds_pi' .- ss[3])

# Generate points
points_yv = LinRange(bounds[1, 1], bounds[1, 2], N)
points_chiv = LinRange(bounds[2, 1], bounds[2, 2], N)
points_piv = LinRange(bounds[3, 1], bounds[3, 2], N)

# Generate the Cartesian product of points_yv, points_chiv, points_piv
coll_ex = collect(product(points_yv, points_chiv, points_piv))

# Convert the result into a matrix (each column is one combination)
coll_ex = hcat([collect(c) for c in coll_ex]...)

# Compute y
y = (2 .* coll_ex .- (bounds[:, 1] .+ bounds[:, 2]) * ones(1, N_ex)) ./ ((bounds[:, 2] .- bounds[:, 1]) * ones(1, N_ex))

# Assuming T_nosmolyak is a function defined elsewhere
TT = T_nosmolyak(y, N, 3)
