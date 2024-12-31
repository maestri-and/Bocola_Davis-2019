###############################################################################
################################## MODEL_SOLUTION.JL #################################

############################## This script solves #############################
################## the benchmark Bocola & Dovis [2019] model ##################

###############################################################################

#------------------------------------------------------------------------------
#       0. Importing libraries and pre-allocated functions and parameters
#------------------------------------------------------------------------------

using LinearAlgebra
# using BenchmarkTools
using GaussQuadrature
using Base.Iterators: product
using DelimitedFiles
using Distances

include("parameters.jl")
include("auxiliary_functions.jl")
include("process.jl")
include("price_schedule_updating.jl")

###############################################################################

#------------------------------------------------------------------------------
# 3  Prepare grids and load initial guess for value functions
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# 3.1 Construct grids for exogenous variables and Chebyshev's polynomials
#------------------------------------------------------------------------------

# Generate points
points_y = LinRange(bounds[1, 1], bounds[1, 2], N)
points_chi = LinRange(bounds[2, 1], bounds[2, 2], N)
points_pi = LinRange(bounds[3, 1], bounds[3, 2], N)

# Generate possible combinations of points
# Create the Cartesian product of points_y, points_chi, and points_pi
coll_points = hcat(product(points_y, points_chi, points_pi)...)
coll_points = hcat([collect(state) for state in coll_points]...)

# 1. Transform coll_points to y_temp
y_temp = transpose((2 .* coll_points .- (bounds[:, 1] .+ bounds[:, 2])) ./ (bounds[:, 2] .- bounds[:, 1]))

# 2. Initialize T for the first iteration
T[:, 1, :] .= 1

# 3. Copy y_temp to the second column of T
T[:, 2, :] .= y_temp

# 4. Apply recurrence relation
for i in 3:N
    for j in 1:state_ex
        T[:, i, j] .= 2 .* y_temp[:, j] .* T[:, i-1, j] .- T[:, i-2, j]
    end
end

# Iterate over each value of j (rows of Tprod)
for j = 1:N_ex
    l = 0
    # Loop over the 3 indices (k, m, aa) in the second and third dimensions of T
    for k = 1:N
        for m = 1:N
            for aa = 1:N
                l += 1  # Increment the column index for Tprod
                # Calculate the product and store it in Tprod[j, l]
                Tprod[j, l] = T[j, k, 1] * T[j, m, 2] * T[j, aa, 3]
            end
        end
    end
end

# Transpose the result
TT = transpose(Tprod)


#------------------------------------------------------------------------------
# 3.2 Construct grid for debt (value function, equidistant points)
#------------------------------------------------------------------------------

# 3.2 Construct grid for debt
debt = LinRange(b_min, b_max, N_b)


#------------------------------------------------------------------------------
# 3.3.  Generate grid for maturity choices
#------------------------------------------------------------------------------

# 3.3 Generate grid for maturity choices
lam_back = LinRange(lam_min, lam_max, N_l)

# Create lam grid (inverse of lam_back, with reverse indexing)
lam = 1 ./ (4 .* reverse(lam_back))


#------------------------------------------------------------------------------
# 3.4   Construct grid for debt choices
#------------------------------------------------------------------------------

# 3.4.1 Construct grid for B1, B2, and B3
B1 = LinRange(b1_min, b1_max, N_b1)
B2 = LinRange(b2_min, b2_max, N_s - N_b1 - N_b3)
B3 = LinRange(b3_min, b3_max, N_b3)

# 3.4.2 Construct Bline by concatenating B1, B2, and B3
Bline = vcat(B1, B2, B3)  # Concatenates B1, B2, and B3 into one vector



#------------------------------------------------------------------------------
# 3.5 Construct grids for exogenous variables in pricing schedule
#------------------------------------------------------------------------------

points_qy = LinRange(bounds[1, 1], bounds[1, 2], N_py)
points_qchi = LinRange(bounds[2, 1], bounds[2, 2], N_pchi)
points_qpi  = LinRange(bounds[3, 1], bounds[3, 2], N_ppi)

# Generate possible combinations of points
# Create the Cartesian product of points_y, points_chi, and points_pi
coll_price = hcat(product(points_qy, points_qchi, points_qpi)...)
coll_price = hcat([collect(state) for state in coll_price]...)

# Normalize coll_price to get A (matching MATLAB normalization)
A = (2 * coll_price .- ((bounds[:, 1] .+ bounds[:, 2]) * ones(1, N_ps))) ./ 
    ((bounds[:, 2] .- bounds[:, 1]) * ones(1, N_ps))

# Normalize coll_ex to get B (assuming coll_ex is given as an input array)
B = (2 * coll_ex[1:3, :] .- ((bounds[:, 1] .+ bounds[:, 2]) * ones(1, N_ex))) ./ 
    ((bounds[:, 2] .- bounds[:, 1]) * ones(1, N_ex))


# Find the nearest neighbor using the Euclidean distance (similar to dsearchn)
# Since A and B are columns of points, you can calculate the distance matrix
distances = pairwise(Euclidean(), A, B)  # Transpose A and B to match dimensions
price_index = argmin(distances, dims=1)    # Find the indices of the closest points in B to A

# price_index will contain the index of the nearest point in B for each column in A

#------------------------------------------------------------------------------
# 3.6 Points and weights for Gauss-Hermite integration
#------------------------------------------------------------------------------

points, weights1 = hermite(N_ghq)
weights          = kron(weights1, weights1)
weights          = kron(weights1, weights)

points = hcat(product(points, points, points)...)
points = hcat([collect(x) for x in points]...)

#------------------------------------------------------------------------------
# 3.7 Steady State
#------------------------------------------------------------------------------

# Steady state values
ss = ss
# Already defined in parameters.jl - Bounds for exogenous state variables


#------------------------------------------------------------------------------
# 3.8 Initial Guess
#------------------------------------------------------------------------------

println("Importing initial guess...")
@elapsed q_old = reshape(readdlm("./data/initial_guess/pricing.txt", ','), N_s, N_l, N_l, N_price)
gamma_a = reshape(readdlm("./data/initial_guess/gamma_a.txt", ','),:)
gamma_p = reshape(readdlm("./data/initial_guess/gamma_p.txt", ','), N_l, N_b, N_ex)
gamma_ck = reshape(readdlm("./data/initial_guess/gamma_ck.txt", ','), N_l, N_b, N_ex)
bprime_old = reshape(readdlm("./data/initial_guess/b_policy.txt", ','), N_l, N_b, N_ex)
lamprime_old = reshape(readdlm("./data/initial_guess/lam_policy.txt", ','), N_l, N_b, N_ex)


#------------------------------------------------------------------------------
# 3.9 Chebyshev's polynomials
#------------------------------------------------------------------------------

InvTT = inv(TT)

#------------------------------------------------------------------------------
# 3.10 Create auxiliary variables for computations
#------------------------------------------------------------------------------

y_y = ss[1,1] .+ coll_points[1, :]
chi_y = ss[2,1] .+ coll_points[2, :]
pi_y = ss[3,1] .+ coll_points[3, :]

extra[1,1] = bounds[1,1]+bounds[1,2]
extra[2,1] = bounds[2,1]+bounds[2,2]
extra[3,1] = bounds[3,1]+bounds[3,2]
extra2[1,1]= bounds[1,2]-bounds[1,1]
extra2[2,1]= bounds[2,2]-bounds[2,1]
extra2[3,1]= bounds[3,2]-bounds[3,1]

# Define the `x_prime` matrix and initialize necessary variables


# Part 1: Loop over N_ex and N_p
for i in 1:N_ex
    for j in 1:N_p
        x_prime = zeros(state_ex, 1)
        x_prime[1, 1] = rho_y * y_y[i] + sqrt(2) * sigma_y * points[1, j] + sqrt(2) * sigma_yk * points[2, j]
        x_prime[2, 1] = mu_chi + rho_chi * chi_y[i] + sqrt(2) * sigma_chi * points[2, j]
        x_prime[3, 1] = exp(pi_star + sqrt(2) * sigma_pi * points[3, j]) / (1 + exp(pi_star + sqrt(2) * sigma_pi * points[3, j]))

        y_prime = (2 * (x_prime .- ss) .- extra) ./ extra2
        y_prime = clamp.(y_prime, -1.0, 1.0)

        y_prime_TP = transpose(y_prime)

        Tgen!(TTT1[i, j, :, :], y_prime_TP, N, state_ex, N_ex)
    end
end

# 1. Populate y_p, chi_p, pi_p based on coll_price and ss
y_p = coll_price[1, :]
chi_p = coll_price[2, :] .+ ss[2, 1]
pi_p = coll_price[3, :] .+ ss[3, 1]

# 2. Loop through i (1 to N_price) and j (1 to N_pq)
for i in 1:N_price
    for j in 1:N_pq
        # 3. Calculate x_prime components based on the given formula
        x_prime = zeros(3, 1)
        x_prime[1, 1] = rho_y * y_p[i] + sqrt(2) * sigma_y * pointsq[1, j] + sqrt(2) * sigma_yk * pointsq[2, j]
        x_prime[2, 1] = mu_chi + rho_chi * chi_p[i] + sqrt(2) * sigma_chi * pointsq[2, j]
        x_prime[3, 1] = exp(pi_star + sqrt(2) * sigma_pi * pointsq[3, j]) / (1 + exp(pi_star + sqrt(2) * sigma_pi * pointsq[3, j]))

        # 4. Calculate y_prime with constraints
        y_prime = (2 .* (x_prime .- ss) .- extra) ./ extra2
        y_prime = clamp.(y_prime, -1.0, 1.0)  # Clamping values between -1 and 1

        # 5. Transpose y_prime
        y_prime_TP = transpose(y_prime)

        # 6. Call the Tgen function with the current values
        Tgen!(TTT2[i, j, :, :], y_prime_TP, N, state_ex, N_ex)
    end
end


#------------------------------------------------------------------------------
# 3.11 Create indexes for computation
#------------------------------------------------------------------------------

N_sl, N_su, s_prime, index_s, 
index_ck, index_ex1, index_ex2 = locator(coll_points, coll_price, bounds, ss, 
    Bline, lam, debt, weightsq, pointsq, index_s, 
    index_ck, index_ex1, index_ex2, N_sl, N_su, s_prime, 
    points_qy, points_qchi, points_qpi, 
    points_y, points_chi, points_pi)

#------------------------------------------------------------------------------
# 4. SOLVE MODEL!
#------------------------------------------------------------------------------

# Initialise variables
q_new       = q_old
bprime      = bprime_old
qeq_old     = 0
lamprime    = lamprime_old
iter        = 1
aa          = 0
ex          = 0
weight_q    = 0 
gamma_pnew = zeros(Float32, N_l, N_b, N_ex)
gamma_cknew = zeros(Float32, N_l, N_b, N_ex)
gamma_anew = zeros(Float32, N_ex)
debt_choice = zeros(Float32, N_l, N_b, N_ex)
maturity_choice = zeros(Float32, N_l, N_b, N_ex) 


# Temporary arrays
value_aut = zeros(Float32, N_ex, 1)
gam_bound = zeros(Float32, N_ex)
value_pay = zeros(Float32, N_ex, N_l, N_b, N_bb, N_l)
value_ck = zeros(Float32, N_l, N_b, N_ex)
maxv = zeros(Float32, N_l, N_b, N_ex)
maxv1 = zeros(Float32, N_ex, N_l, N_b, N_bb)
lamprime1 = zeros(Float32, N_l, N_b, N_ex, N_bb)

# Step 2.2: Generate points for exogenous shocks
y_y = ss[1,1] .+ coll_points[1, :]
chi_y = ss[2,1] .+ coll_points[2, :]
pi_y = ss[3,1] .+ coll_points[3, :]

# Step 2.3: Generate output costs of default
d_0 = (d0 - d1 * exp(bounds[1, 1])) / (1 - exp(bounds[1, 1]))
d_1 = d1 - d_0
d = d_0 .+ d_1 .* exp.(y_y)

# Step 3: Update value of defaulting
@threads for i in 1:N_ex
    # Re-initialize variables for each thread
    TT_def = zeros(Float32, N_p, 1)
    TT_nodef = zeros(Float32, N_p, 1)
    TT_ck = zeros(Float32, N_p, 1)
    gam_prov = zeros(Float32, 1, N_ex)
    cont_fund = 0.0
    cont_fund_u = 0.0
    spend = 0.0

    # Inner loop over j
    for j in 1:N_p
        # Using reshape inside the thread to assign values to gam_prov
        gam_prov[:] .= reshape(gamma_p[1, 1, :], N_ex)
        cont_fund = 0.0
        cont_fund_u = 0.0

        # Loop over k
        for k in 1:N_ex
            cont_fund += gam_prov[1,k] * TTT1[i,j,k,1]
            cont_fund_u += gamma_a[k] * TTT1[i,j,k,1]
        end

        # Assign results to TT_nodef and TT_def
        TT_nodef[j, 1] = cont_fund
        TT_def[j, 1] = cont_fund_u
    end

    # Calculate final result for the value_aut vector
    cont_fund = sum(weights .* (psi .* max.(TT_nodef, TT_def) .+ (1 .- psi) .* TT_def))
    spend = max(exp(y_y[i]) * (1 - d[i]) - g_star, 0.005)
    value_aut[i, 1] = (spend^(1 - sigma) - 1) / (1 - sigma) + beta * π^(-1.5) * cont_fund
end

gamma_anew .= InvTT * value_aut

gam_bound .= minimum(value_aut) - 0.75

# Step 4: Update value of repaying if lenders do not rollover - Slow section
# Vectorized faster version    
@threads for j in 1:N_ex
    for m in 1:N_l
        for i in 1:N_b
            # Initialize variables
            TT_def = zeros(Float32, N_p, 1)
            TT_nodef = zeros(Float32, N_p, 1)
            TT_ck = zeros(Float32, N_p, 1)
            safe = zeros(Float32, N_p, 1)
            safe .= 0.0

            # Vectorized calculation of TT_nodef, TT_ck, and safe
            for k in 1:N_p
                idx_ck = index_ck[i, m, 1]
                TTT1_jk = TTT1[j, k, :, 1]  # Extract slice for easier access

                # Vectorized accumulation of cont_fund_l, cont_fund_u, cont_fund_lck, cont_fund_uck
                if i > 1
                    cont_fund_l = sum(gamma_p[m, idx_ck, 1:N_ex] .* TTT1_jk)
                    cont_fund_u = sum(gamma_p[m, idx_ck + 1, 1:N_ex] .* TTT1_jk)
                    cont_fund_lck = sum(gamma_ck[m, idx_ck, 1:N_ex] .* TTT1_jk)
                    cont_fund_uck = sum(gamma_ck[m, idx_ck + 1, 1:N_ex] .* TTT1_jk)

                    # Precompute denominator to optimize calculation
                    denom = debt[idx_ck + 1, 1] - debt[idx_ck, 1]

                    TT_nodef[k, 1] = cont_fund_l * ((debt[idx_ck + 1, 1] - debt[i, 1] * (1 - lam[m, 1])) / denom) +
                                    cont_fund_u * ((debt[i, 1] * (1 - lam[m, 1]) - debt[idx_ck, 1]) / denom)

                    TT_ck[k, 1] = cont_fund_lck * ((debt[idx_ck + 1, 1] - debt[i, 1] * (1 - lam[m, 1])) / denom) +
                                    cont_fund_uck * ((debt[i, 1] * (1 - lam[m, 1]) - debt[idx_ck, 1]) / denom)
                end

                # Handle the case when i == 1 (no need for loops)
                if i == 1
                    cont_fund_u = sum(gamma_p[m, i, 1:N_ex] .* TTT1_jk)
                    cont_fund_uck = sum(gamma_ck[m, i, 1:N_ex] .* TTT1_jk)

                    TT_nodef[k, 1] = cont_fund_u
                    TT_ck[k, 1] = cont_fund_uck
                end

                # Vectorized calculation of TT_def
                cont_fund = sum(gamma_a .* TTT1_jk)
                TT_def[k, 1] = cont_fund

                # Vectorized safe calculation (comparison)
                safe[k, 1] = TT_ck[k, 1] >= TT_def[k, 1] ? 1.0 : 0.0
            end

            # Vectorized calculation for total cont_fund using broadcasting
            cont_fund = sum(weights .* (safe .* max.(TT_def, TT_nodef) .+ 
                                        (1 .- safe) .* (pi_y[j] .* TT_def .+ 
                                        (1 - pi_y[j]) .* max.(TT_def, TT_nodef))))

            # Vectorized spend calculation
            spend = max(exp(y_y[j]) - debt[i, 1] * lam[m, 1] - g_star, 0.005)

            # Final value calculation
            value_ck[m, i, j] = (spend^(1 - sigma) - 1) / (1 - sigma) + beta * π^(-1.5) * cont_fund
        end
    end
end

# Unparallelised
for l in 1:N_l
    for k in 1:N_b
        gam_prov = zeros(N_ex)  # Allocating gam_prov
        gam_prov .= value_ck[l, k, :]  # Assuming value_ck is pre-defined
        
        # Apply max constraint
        for i in 1:N_ex
            gam_prov[i] = max(gam_bound[i], gam_prov[i])
        end
        
        gam_prov .=  InvTT * gam_prov
        gamma_cknew[l, k, :] .= gam_prov  # Store result
    end
end
