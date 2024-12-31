###############################################################################
################################# PROCESS.JL ##################################

################## This script defines the functions to solve #################
################## the benchmark Bocola & Dovis [2019] model ##################

###############################################################################

#------------------------------------------------------------------------------
#              0. Importing libraries and pre-allocated parameters
#------------------------------------------------------------------------------

using LinearAlgebra
# using SharedVector
using BenchmarkTools
using Base.Iterators: product
using Base.Threads

include("parameters.jl")

###############################################################################

#------------------------------------------------------------------------------
#                               1.  VFI updating 
#------------------------------------------------------------------------------

###############################################################################



function update_values!(gamma_pnew, gamma_anew, gamma_cknew, bprime, 
                        lamprime, debt_choice, maturity_choice, 
                        q_old, gamma_p, gamma_a, gamma_ck, 
                        coll_points, InvTT, bounds, ss, coll_price, 
                        price_index, Bline, debt, lam, weights, 
                        points, TTT1, index_s, index_ck, N_sl, N_su)

    # This function updates the value functions

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
    
    # 5 Update value of repaying - Slow section
    
    value_pay .= -4000
    maxv .= -10000
    maxv1 .= 0

    # Pre-allocate variables to improve performances
    cont_fund_u = 0
    cont_fund_l = 0
    cont_fund_uck = 0
    cont_fund_lck = 0
    kk = 0
    # Preallocate arrays to store values for each `l`
    debt_l_plus_1 = 0
    debt_l = 0

    debt_diff = 0
    debt_sum = 0
    debt_l_plus_1_minus_Bline = 0
    Bline_l_minus_debt_l = 0


    # First (slower) version - to be deleted
    @time @threads for i in 1:N_ex
        start_test = time()
        # TO BE DELETED - Checking time 
        if i in (2, 5, 10, 20, 50, 100)
            clock = time()
            toprint = clock - start_test
            println("Iteration m, i : ($m, $i)")
            println("Seconds from start : $toprint")
            println("Minutes from start : $(toprint/60)")
        end
        # Preallocate arrays outside the loops
        TT_def = zeros(N_p)
        TT_nodef = zeros(N_p)
        TT_ck = zeros(N_p)
        safe = zeros(N_p)


        # Use @threads to parallelize the outermost loop
        start_test = time()
        @time @threads for k in 1:N_b
            # TO BE DELETED - Checking time
            if k == 1
                println("Starting test...")
            end 
            if k in (2, 5, 10, 20, 40, 80)
                clock = time()
                toprint = clock - start_test
                println("Iteration k : ($k)")
                println("Seconds from start : $toprint")
                println("Minutes from start : $(toprint/60)")
            end
            kk = 0
            for l in N_sl[k, 1]:N_su[k, 1]
                kk += 1
                # Initialize safe and other variables inside the loop
                safe .= 0
                TT_def .= 0
                TT_nodef .= 0
                TT_ck .= 0
                
                # Precompute differences that are used in multiple places
                debt_l_plus_1 = debt[index_s[l, 1] + 1, 1]
                debt_l = debt[index_s[l, 1], 1]
                Bline_l = Bline[l]

                debt_diff = debt_l_plus_1 - debt_l
                debt_sum = debt_l_plus_1 + debt_l
                debt_l_plus_1_minus_Bline_l = debt_l_plus_1 - Bline_l
                Bline_l_minus_debt_l = Bline_l - debt_l

                # Contingency variables precomputed for each j
                for o in 1:N_l
                    # Reset contingency variables at the start of each o loop
                    cont_fund_u = 0.0
                    cont_fund_l = 0.0
                    cont_fund_uck = 0.0
                    cont_fund_lck = 0.0
    
                    for j in 1:N_p
                        # Accumulate contingency funds for each m
                        for m in 1:N_ex
                            cont_fund_l += gamma_p[o, index_s[l, 1], m] * TTT1[i, j, m, 1]
                            cont_fund_u += gamma_p[o, index_s[l, 1] + 1, m] * TTT1[i, j, m, 1]
                            cont_fund_lck += gamma_ck[o, index_s[l, 1], m] * TTT1[i, j, m, 1]
                            cont_fund_uck += gamma_ck[o, index_s[l, 1] + 1, m] * TTT1[i, j, m, 1]
                        end

                        # Compute TT_nodef and TT_ck based on contingency variables
                        TT_nodef[j] = cont_fund_l * (debt_l_plus_1_minus_Bline_l / debt_diff) +
                        cont_fund_u * (Bline_l_minus_debt_l / debt_diff)

                        TT_ck[j] = cont_fund_lck * (debt_l_plus_1_minus_Bline_l / debt_diff) +
                        cont_fund_uck * (Bline_l_minus_debt_l / debt_diff)
    
                        # # Compute TT_nodef and TT_ck based on contingency variables, as in Fortran
                        # TT_nodef[j] = cont_fund_l * ((debt[index_s[l, 1] + 1, 1] - Bline[l, 1]) / (debt[index_s[l, 1] + 1, 1] - debt[index_s[l, 1], 1])) +
                        # cont_fund_u * ((Bline[l, 1] - debt[index_s[l, 1], 1]) / (debt[index_s[l, 1] + 1, 1] - debt[index_s[l, 1], 1]))

                        # TT_ck[j] = cont_fund_lck * ((debt[index_s[l, 1] + 1, 1] - Bline[l, 1]) / (debt[index_s[l, 1] + 1, 1] - debt[index_s[l, 1], 1])) +
                        # cont_fund_uck * ((Bline[l, 1] - debt[index_s[l, 1], 1]) / (debt[index_s[l, 1] + 1, 1] - debt[index_s[l, 1], 1]))

                        # Calculate TT_def for each j
                        cont_fund = 0
                        for m in 1:N_ex
                            cont_fund += gamma_a[m] * TTT1[i, j, m, 1]
                        end
                        TT_def[j] = cont_fund
                        safe[j] = TT_ck[j] .>= TT_def[j]  # Broadcasting comparison
                    end
    
                    # Perform weighted sums and update values for each p
                    for p in 1:N_l
                        cont_fund = weights' * (safe .* max.(TT_def, TT_nodef) .+
                                                (1 .- safe) .* (pi_y[i] .* TT_def .+ 
                                                (1 .- pi_y[i]) .* max.(TT_def, TT_nodef)))
    
                        # Avoid unnecessary recalculations by storing the common terms
                        common_spend = max(exp(y_y[i]) - lam[p, 1] * debt[k, 1] + 
                                            q_old[l, o, o, price_index[1, i]] * Bline[l, 1] - 
                                            q_old[l, o, p, price_index[1, i]] * (1 .- lam[p, 1]) * debt[k, 1] - 
                                            g_star, 0.005)
    
                        value_pay[i, p, k, kk, o] = (common_spend^(1 - sigma) - 1) / (1 - sigma) + 
                                                    (beta * π^(-1.5)) * cont_fund[1] - 
                                                    alpha * ((1 / (4 * lam[o, 1])) - d_upper)^2
                    end
                end
            end
        end
    end
        
    # for k in 1:N_b
        #     kk = 0
        #     for l in N_sl[k, 1]:N_su[k, 1]
        #         kk += 1
        #         for o in 1:N_l
        #             TT_def = zeros(N_p)
        #             TT_nodef = zeros(N_p)
        #             TT_ck = zeros(N_p)
        #             safe = zeros(N_p)
                    
        #             # Initialize safe
        #             safe .= 0
    
        #             for j in 1:N_p
        #                 cont_fund_u = 0
        #                 cont_fund_l = 0
        #                 cont_fund_uck = 0
        #                 cont_fund_lck = 0
                        
        #                 for m in 1:N_ex
        #                     cont_fund_l += gamma_p[o, index_s[l, 1], m] * TTT1[i, j, m, 1]
        #                     cont_fund_u += gamma_p[o, index_s[l, 1] + 1, m] * TTT1[i, j, m, 1]
        #                     cont_fund_lck += gamma_ck[o, index_s[l, 1], m] * TTT1[i, j, m, 1]
        #                     cont_fund_uck += gamma_ck[o, index_s[l, 1] + 1, m] * TTT1[i, j, m, 1]
        #                 end
                        
        #                 TT_nodef[j] = cont_fund_l * ((debt[index_s[l, 1] + 1, 1] - Bline[l, 1]) / 
        #                                              (debt[index_s[l, 1] + 1, 1] - debt[index_s[l, 1], 1])) + 
        #                                              cont_fund_u * ((Bline[l, 1] - debt[index_s[l, 1], 1]) / 
        #                                              (debt[index_s[l, 1] + 1, 1] - debt[index_s[l, 1], 1]))
        #                 TT_ck[j] = cont_fund_lck * ((debt[index_s[l, 1] + 1, 1] - Bline[l, 1]) / 
        #                                              (debt[index_s[l, 1] + 1, 1] - debt[index_s[l, 1], 1])) + 
        #                                              cont_fund_uck * ((Bline[l, 1] - debt[index_s[l, 1], 1]) / 
        #                                              (debt[index_s[l, 1] + 1, 1] - debt[index_s[l, 1], 1]))
    
        #                 cont_fund = 0
        #                 for m in 1:N_ex
        #                     cont_fund += gamma_a[m] * TTT1[i, j, m, 1]
        #                 end
        #                 TT_def[j] = cont_fund
        #                 safe[j] = TT_ck[j] .>= TT_def[j]  # Using broadcasting to compare element-wise
        #             end
                    
        #             for p in 1:N_l
        #                 # possible issue
        #                 cont_fund = weights' * (safe .* max.(TT_def, TT_nodef) .+ 
        #                 (1 .- safe) .* (pi_y[i] .* TT_def .+ (1 .- pi_y[i]) .* max.(TT_def, TT_nodef)))

        #                 spend = max(exp(y_y[i]) - lam[p, 1] * debt[k, 1] + q_old[l, o, o, price_index[1, i]] * Bline[l, 1] - 
        #                             q_old[l, o, p, price_index[1, i]] * (1 .- lam[p, 1]) * debt[k, 1] - g_star, 0.005)
        #                 value_pay[i, p, k, kk, o] = (spend^(1 - sigma) - 1) / (1 - sigma) + 
        #                                                 (beta * π^(-1.5)) * cont_fund[1] - alpha * ((1 / (4 * lam[o, 1])) - d_upper)^2
        #             end
        #         end
        #         for p in 1:N_l
        #             maxv1[i, p, k, kk] = maximum(value_pay[i, p, k, kk, :])
        #             maxl = argmax(value_pay[i, p, k, kk, :])
        #             if maxv[p, k, i] < maxv1[i, p, k, kk]
        #                 maxv[p, k, i] = maxv1[i, p, k, kk]
        #                 bprime[p, k, i] = l
        #                 lamprime[p, k, i] = maxl[1]
        #                 debt_choice[p, k, i] = Bline[l, 1]
        #                 maturity_choice[p, k, i] = lam[maxl[1], 1]
        #             end
        #         end
        #     end
        #     aa = maximum(maxv[:, k, i])
        #     if aa < gam_bound[1, i]
        #         for n in k:N_b
        #             maxv[:, n, i] .= gam_bound[1, i]
        #             bprime[:, n, i] .= N_s
        #             lamprime[:, n, i] .= N_l
        #             debt_choice[:, n, i] .= Bline[N_s, 1]
        #             maturity_choice[:, n, i] .= lam[N_l, 1]
        #         end
        #         break
        #     end
        # end
    # end
    
    # Unparallelised
    for l in 1:N_l
        for k in 1:N_b
            gam_prov = zeros(N_ex)
            gam_prov .= maxv[l, k, :]
            
            for i in 1:N_ex
                gam_prov[i] = max(gam_bound[i], gam_prov[i])
            end
            
            gam_prov .= InvTT * gam_prov
            gamma_pnew[l, k, :] .= gam_prov
        end
    end
end


###############################################################################


###############################################################################

#------------------------------------------------------------------------------
#                                   4.  LOCATOR
#------------------------------------------------------------------------------

###############################################################################




function locator(coll_points, coll_price, bounds, ss, 
    Bline, lam, debt, weightsq, pointsq, index_s, 
    index_ck, index_ex1, index_ex2, N_sl, N_su, s_prime, 
    points_qy, points_qchi, points_qpi, 
    points_y, points_chi, points_pi)
    #---------------------------------------------
    # 0. Define variables
    #---------------------------------------------

    # Model variables
    # i, j, k = 0, 0, 0
    temp_ex1 = zeros(Int, 1)
    temp_ex2 = zeros(Int, 1)
    index_ss = zeros(Int, 2)
    dd = 0.0f0

    ones_Np = ones(Float32, N_price)
    ones_Nc = ones(Float32, N_ex)
    
    x_prime = zeros(Float32, size(coll_points)...)
    y_prime = zeros(Float32, size(coll_points)...)
   
    coll_transformed = (2 * (coll_points) .- extra * ones_Nc') ./ (extra2 * ones_Nc')
    coll_transformed1 = (2 * (coll_price) .- extra * ones_Np') ./ (extra2 * ones_Np')

    # Handle debt indexing
    for i in 1:N_b
        for j in 1:N_l
            # Use searchsorted to find the insertion index in the debt array
            idx = searchsorted(debt[:, 1], debt[i, 1] * (1 - lam[j, 1]))
    
            # Extract the first index if it is a range
            idx = first(idx)

            # If the index is out of bounds, adjust it to N_b
            if idx == N_b + 1
                idx = N_b
            end

            if idx == N_b 
                idx = N_b - 1
            end
    
            # Store the result back in index_ck
            index_ck[i, j, 1] = idx
        end
    end

    # Handle index for state variables
    for i in 1:N_s
        # Find the index using searchsorted
        idx = searchsorted(debt[:, 1], Bline[i, 1])
        
        # Extract the first element of the index range, if necessary
        idx = first(idx)
        
        # Adjust the index if it is out of bounds
        if idx == N_b + 1
            idx = N_b
        end

        if idx == N_b 
            idx = N_b - 1
        end
        
        # Store the result back in index_s
        index_s[i, 1] = idx
    end

    # Generate pricing schedule
    for j in 1:N_price
        x_prime .= 0  # Clear x_prime before filling it
        y_prime .= 0  # Clear y_prime before filling it
        for k in 1:N_pq
            x_prime[1, 1] = rho_y * y_p[j] + 2 ^ 0.5 * sigma_y * pointsq[1, k] + 2 ^ 0.5 * sigma_yk * pointsq[2, k]
            x_prime[2, 1] = mu_chi + rho_chi * chi_p[j] + 2 ^ 0.5 * sigma_chi * pointsq[2, k]
            x_prime[3, 1] = exp(pi_star + 2 ^ 0.5 * sigma_pi * pointsq[3, k]) / (1 + exp(pi_star + 2 ^ 0.5 * sigma_pi * pointsq[3, k]))

            y_prime = (2 * (x_prime .- ss) .- extra) ./ extra2
            y_prime = max.(y_prime, -1.0f0)
            y_prime = min.(y_prime, 1.0f0)

            s_prime[1, j, k] = min(max(x_prime[1, 1], points_qy[1, 1]), points_qy[N_py, 1])
            s_prime[2, j, k] = min(max(x_prime[2, 1] .- ss[2, 1], points_qchi[1, 1]), points_qchi[N_pchi, 1])
            s_prime[3, j, k] = min(max(x_prime[3, 1] .- ss[3, 1], points_qpi[1, 1]), points_qpi[N_ppi, 1])

            # Bisecting for state variables
            index_ex2[1, j, k] = first(searchsorted(points_qy[:, 1], s_prime[1, j, k]))
            index_ex2[2, j, k] = first(searchsorted(points_qchi[:, 1], s_prime[2, j, k]))
            index_ex2[3, j, k] = first(searchsorted(points_qpi[:, 1], s_prime[3, j, k]))

            index_ex1[1, j, k] = first(searchsorted(points_y[:, 1], s_prime[1, j, k]))
            index_ex1[2, j, k] = first(searchsorted(points_chi[:, 1], s_prime[2, j, k]))
            index_ex1[3, j, k] = first(searchsorted(points_pi[:, 1], s_prime[3, j, k]))
        end
    end

    # Find the indices of lower and upper bounds for debt
    N_sl .= 1
    for k in 1:N_b
        dd = debt[k, 1]
        index_ss = argmin(abs.(dd .- lwb .- Bline))
        N_sl[k, 1] = index_ss[1]
        index_ss = argmin(abs.(dd .+ uwb .- Bline))
        N_su[k, 1] = index_ss[1]
    end

    return N_sl, N_su, s_prime, index_s, index_ck, index_ex1, index_ex2
end
