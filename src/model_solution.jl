###############################################################################
################################## PROCESS.JL #################################

############################## This script solves #############################
################## the benchmark Bocola & Dovis [2019] model ##################

###############################################################################

#------------------------------------------------------------------------------
#       0. Importing libraries and pre-allocated functions and parameters
#------------------------------------------------------------------------------

using LinearAlgebra
using BenchmarkTools
using Printf
using Random
# using IterTools - TO BE DELETED
using GaussQuadrature
using Base.Iterators: product

include("parameters.jl")
include("auxiliary_functions.jl")
include("process.jl")


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

# gamma_a
# gamma_p
# gamma_ck
# q_old
# bprime_old
# lamprime_old

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

q_new       = q_old
bprime      = bprime_old
qeq_old     = 0
lamprime    = lamprime_old
iter        = 1
start       = omp_get_wtime ( )
aa          = 0
ex          = 0
weight_q    = 0  

while (aa==0 && iter<max_iter)

    update_values!(gamma_pnew, gamma_anew, gamma_cknew, 
    bprime, lamprime, debt_choice, maturity_choice, 
    q_old, gamma_p, gamma_a, gamma_ck, coll_points, 
    InvTT, bounds, ss, coll_price, price_index, 
    Bline, debt, lam, weights, points, TTT1, 
    index_s, index_ck, N_sl, N_su)

    update_price!(q_new, q_old, gamma_p, gamma_a, gamma_ck, 
    coll_points, coll_price, bounds, ss, price_index, 
    Bline, lam, debt, weightsq, pointsq, 
    bprime_old, lamprime_old, TTT2, 
    index_s, index_ex1, index_ex2, s_prime, 
    points_qy, points_qchi, points_qpi, 
    points_y, points_chi, points_pi)

    if (iter==start_conv)
        weight_q = convergence_q 
    end

    if (iter>=start_conv)  
        weight_q = min(weight_q+(0.001/100),0.9995)
    end 

    # Initialize or reset `start` before the loop if needed
    start = time()

    # Loop body 
    for iter in 1:max_iterations  # assuming a max iteration limit

        # Update values with matrix multiplication
        for j in 1:N_b
            value_paynew[:, :, j]  .= matmul(reshape(gamma_pnew[:, j, :], N_l, N_ex), TT)
            value_payold[:, :, j]  .= matmul(reshape(gamma_p[:, j, :], N_l, N_ex), TT)
            value_cknew[:, :, j]   .= matmul(reshape(gamma_cknew[:, j, :], N_l, N_ex), TT)
            value_ckold[:, :, j]   .= matmul(reshape(gamma_ck[:, j, :], N_l, N_ex), TT)
        end

        # Loop for updating qeq_new
        for l in 1:N_l
            for j in 1:N_b
                for k in 1:N_ex
                    qeq_new[l, k, j] = q_new[bprime_old[l, j, k], lamprime_old[l, j, k], l, price_index[1, k]]
                end
            end
        end

        # Calculate max differences
        maxdiff_pay = maximum(abs(log.(value_paynew ./ value_payold)))
        maxdiff_def = maximum(abs(log.(matmul(gamma_a, TT) ./ matmul(gamma_anew, TT))))
        maxdiff_ck  = maximum(abs(log.(value_cknew ./ value_ckold)))
        maxdiff_q   = maximum(abs(qeq_old - qeq_new) .^ 2)

        # measure time for the iteration
        finish = time() - start

        # Print results
        println("Iteration                ", iter)
        println("Norm, Value of repaying  ", maxdiff_pay)
        println("Norm, Value of defaulting", maxdiff_def)
        println("Norm, Value of ck        ", maxdiff_ck)
        println("Norm, bond prices        ", maxdiff_q)
        println("Time for iteration       ", finish)

        # Update start time for the next iteration
        start = time()
    end

    # Update variables for next iteration
    q_old = weight_q * q_old + (1 - weight_q) * q_new
    qeq_old = qeq_new
    bprime_old = bprime
    lamprime_old = lamprime
    debt_choice_old = debt_choice
    maturity_choice_old = maturity_choice
    gamma_p = convergence_v * gamma_p + (1 - convergence_v) * gamma_pnew
    gamma_ck = convergence_v * gamma_ck + (1 - convergence_v) * gamma_cknew
    gamma_a = convergence_v * gamma_a + (1 - convergence_v) * gamma_anew
    iter += 1  # Increment the iteration counter

    # Convergence check
    if (iter > 2 && 
        maxdiff_def <= tolerance && 
        maxdiff_pay <= tolerance && 
        maxdiff_ck <= tolerance && 
        maxdiff_q <= tolerance)

        aa = 1  # Convergence achieved, exit loop
    end
end




