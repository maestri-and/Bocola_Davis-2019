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
using Printf
using Random
using Base.Iterators: product
using Base.Threads

include("parameters.jl")

###############################################################################

#------------------------------------------------------------------------------
#                               1.  VFI updating 
#------------------------------------------------------------------------------

###############################################################################



function update_values!(gamma_pnew, gamma_anew, gamma_cknew, bprime, lamprime, debt_choice, maturity_choice, q_old, gamma_p, gamma_a, gamma_ck, coll_points, InvTT, bounds, ss, coll_price, price_index, 
                        Bline, debt, lam, weights, points, TTT1, index_s, index_ck, N_sl, N_su)
    # This function updates the value functions

    # Temporary arrays
    value_aut = zeros(Float32, N_ex, 1)
    gam_bound = zeros(Float32, N_ex)
    value_pay = fill(-4000, (N_ex, N_l, N_b, N_bb, N_l))
    value_ck = zeros(Float32, N_l, N_b, N_ex)
    maxv = zeros(Float32, N_l, N_b, N_ex)
    maxv1 = zeros(Float32, N_ex, N_l, N_b, N_bb)
    lamprime1 = zeros(Float32, N_l, N_b, N_ex, N_bb)

    # Step 2.2: Generate points for exogenous shocks
    for i in 1:N_ex
        y_y[i] = ss[1,1] + coll_points[1,i]
        chi_y[i] = ss[2,1] + coll_points[2,i]
        pi_y[i] = ss[3,1] + coll_points[3,i]
    end

    # Step 2.3: Generate output costs of default
    d_0 = (d0 - d1 * exp(bounds[1, 1])) / (1 - exp(bounds[1, 1]))
    d_1 = d1 - d_0
    d = d_0 .+ d_1 .* exp.(y_y)

    # Step 3: Update value of defaulting
    Threads.@threads for i in 1:N_ex
        TT_def = zeros(Float32, N_p, 1)
        TT_nodef = zeros(Float32, N_p, 1)
        TT_ck = zeros(Float32, N_p, 1)
        gam_prov = zeros(Float32, 1, N_ex)
        cont_fund = 0.0
        cont_fund_u = 0.0
        spend = 0.0

        for j in 1:N_p
            gam_prov[:] .= reshape(gamma_p[1, 1, :], N_ex)
            cont_fund = 0.0
            cont_fund_u = 0.0
            for k in 1:N_ex
                cont_fund += gam_prov[1,k] * TTT1[i,j,k,1]
                cont_fund_u += gamma_a[1,k] * TTT1[i,j,k,1]
            end

            TT_nodef[j, 1] = cont_fund
            TT_def[j, 1] = cont_fund_u
        end

        cont_fund = sum(weights .* (psi .* max.(TT_nodef, TT_def) .+ (1 .- psi) .* TT_def))
        spend = max(exp(y_y[i]) * (1 - d[i]) - g_star, 0.005)
        value_aut[i, 1] = (spend^(1 - sigma) - 1) / (1 - sigma) + beta * π^(-1.5) * cont_fund
    end

    gamma_anew .= transpose(value_aut) * InvTT

    gam_bound .= minimum(value_aut) - 0.75

    # Step 4: Update value of repaying if lenders do not rollover
    Threads.@threads for j in 1:N_ex
        for m in 1:N_l
            for i in 1:N_b
                TT_def = zeros(Float32, N_p, 1)
                TT_nodef = zeros(Float32, N_p, 1)
                TT_ck = zeros(Float32, N_p, 1)
                safe = zeros(Float32, N_p, 1)
                safe .= 0.0

                for k in 1:N_p
                    if i > 1
                        cont_fund_u = 0.0
                        cont_fund_l = 0.0
                        cont_fund_uck = 0.0
                        cont_fund_lck = 0.0
                        for l in 1:N_ex
                            cont_fund_l += gamma_p[m, index_ck[i,m,1], l] * TTT1[j,k,l,1]
                            cont_fund_u += gamma_p[m, index_ck[i,m,1]+1, l] * TTT1[j,k,l,1]
                            cont_fund_lck += gamma_ck[m, index_ck[i,m,1], l] * TTT1[j,k,l,1]
                            cont_fund_uck += gamma_ck[m, index_ck[i,m,1]+1, l] * TTT1[j,k,l,1]
                        end

                        TT_nodef[k,1] = cont_fund_l * ((debt[index_ck[i,m,1]+1,1] - debt[i,1]*(1 - lam[m,1])) / (debt[index_ck[i,m,1]+1,1] - debt[index_ck[i,m,1],1])) +
                                        cont_fund_u * ((debt[i,1]*(1 - lam[m,1]) - debt[index_ck[i,m,1],1]) / (debt[index_ck[i,m,1]+1,1] - debt[index_ck[i,m,1],1]))
                        TT_ck[k,1] = cont_fund_lck * ((debt[index_ck[i,m,1]+1,1] - debt[i,1]*(1 - lam[m,1])) / (debt[index_ck[i,m,1]+1,1] - debt[index_ck[i,m,1],1])) +
                                        cont_fund_uck * ((debt[i,1]*(1 - lam[m,1]) - debt[index_ck[i,m,1],1]) / (debt[index_ck[i,m,1]+1,1] - debt[index_ck[i,m,1],1]))
                    end

                    if i == 1
                        cont_fund_u = 0.0
                        cont_fund_uck = 0.0
                        for l in 1:N_ex
                            cont_fund_u += gamma_p[m, i, l] * TTT1[j,k,l,1]
                            cont_fund_uck += gamma_ck[m, i, l] * TTT1[j,k,l,1]
                        end
                        TT_nodef[k,1] = cont_fund_u
                        TT_ck[k,1] = cont_fund_uck
                    end

                    cont_fund = 0.0
                    for l in 1:N_ex
                        cont_fund += gamma_a[1,l] * TTT1[j,k,l,1]
                    end
                    TT_def[k,1] = cont_fund
                    safe[k,1] = TT_ck[k,1] >= TT_def[k,1] ? 1.0 : 0.0
                end

                cont_fund = sum(weights .* (safe .* max.(TT_def, TT_nodef) .+ (1 .- safe) .* (pi_y[j] .* TT_def .+ (1 - pi_y[j]) .* max.(TT_def, TT_nodef))))
                spend = max(exp(y_y[j]) - debt[i,1] * lam[m,1] - g_star, 0.005)
                value_ck[m,i,j] = (spend^(1 - sigma) - 1) / (1 - sigma) + beta * π^(-1.5) * cont_fund
            end
        end
    end
    
    @threads for l in 1:N_l
        for k in 1:N_b
            gam_prov = zeros(N_ex)  # Allocating gam_prov
            gam_prov .= value_ck[l, k, :]  # Assuming value_ck is pre-defined
            
            # Apply max constraint
            for i in 1:N_ex
                gam_prov[i] = max(gam_bound[i], gam_prov[i])
            end
            
            gam_prov .= gam_prov * InvTT  # Matrix multiplication
            gamma_cknew[l, k, :] .= gam_prov  # Store result
        end
    end
    
    # 5 Update value of repaying
    
    value_pay = -4000
    maxv = -10000
    maxv1 = 0
    
    @threads for i in 1:N_ex
        for k in 1:N_b
            kk = 0
            for l in N_sl[k, 1]:N_su[k, 1]
                kk += 1
                for o in 1:N_l
                    TT_def = zeros(N_p)
                    TT_nodef = zeros(N_p)
                    TT_ck = zeros(N_p)
                    safe = zeros(N_p)
                    
                    # Initialize safe
                    safe .= 0
    
                    for j in 1:N_p
                        cont_fund_u = 0
                        cont_fund_l = 0
                        cont_fund_uck = 0
                        cont_fund_lck = 0
                        
                        for m in 1:N_ex
                            cont_fund_l += gamma_p[o, index_s[l, 1], m] * TTT1[i, j, m, 1]
                            cont_fund_u += gamma_p[o, index_s[l, 1] + 1, m] * TTT1[i, j, m, 1]
                            cont_fund_lck += gamma_ck[o, index_s[l, 1], m] * TTT1[i, j, m, 1]
                            cont_fund_uck += gamma_ck[o, index_s[l, 1] + 1, m] * TTT1[i, j, m, 1]
                        end
                        
                        TT_nodef[j] = cont_fund_l * ((debt[index_s[l, 1] + 1, 1] - Bline[l, 1]) / 
                                                     (debt[index_s[l, 1] + 1, 1] - debt[index_s[l, 1], 1])) + 
                                                     cont_fund_u * ((Bline[l, 1] - debt[index_s[l, 1], 1]) / 
                                                     (debt[index_s[l, 1] + 1, 1] - debt[index_s[l, 1], 1]))
                        TT_ck[j] = cont_fund_lck * ((debt[index_s[l, 1] + 1, 1] - Bline[l, 1]) / 
                                                     (debt[index_s[l, 1] + 1, 1] - debt[index_s[l, 1], 1])) + 
                                                     cont_fund_uck * ((Bline[l, 1] - debt[index_s[l, 1], 1]) / 
                                                     (debt[index_s[l, 1] + 1, 1] - debt[index_s[l, 1], 1]))
    
                        cont_fund = 0
                        for m in 1:N_ex
                            cont_fund += gamma_a[1, m] * TTT1[i, j, m, 1]
                        end
                        TT_def[j] = cont_fund
                        safe[j] = TT_ck[j] .>= TT_def[j]  # Using broadcasting to compare element-wise
                    end
                    
                    for p in 1:N_l
                        cont_fund .= matmul(weights, safe .* max.(TT_def, TT_nodef) .+ 
                                              (1 .- safe) .* (pi_y[1, i] .* TT_def .+ (1 .- pi_y[1, i]) .* max.(TT_def, TT_nodef)))
                        spend = max(exp(y_y[1, i]) - lam[p, 1] * debt[k, 1] + q_old[l, o, o, price_index[1, i]] * Bline[l, 1] - 
                                    q_old[l, o, p, price_index[1, i]] * (1 .- lam[p, 1]) * debt[k, 1] - g_star, 0.005)
                        value_pay[i, p, k, kk, o] = (spend^(1 - sigma) - 1) / (1 - sigma) + 
                                                     (beta * π^(-1.5)) * cont_fund[1] - alpha * ((1 / (4 * lam[o, 1])) - d_upper)^2
                    end
                end
                for p in 1:N_l
                    maxv1[i, p, k, kk] = maximum(value_pay[i, p, k, kk, :])
                    maxl = argmax(value_pay[i, p, k, kk, :])
                    if maxv[p, k, i] < maxv1[i, p, k, kk]
                        maxv[p, k, i] = maxv1[i, p, k, kk]
                        bprime[p, k, i] = l
                        lamprime[p, k, i] = maxl[1]
                        debt_choice[p, k, i] = Bline[l, 1]
                        maturity_choice[p, k, i] = lam[maxl[1], 1]
                    end
                end
            end
            aa = maximum(maxv[:, k, i])
            if aa < gam_bound[1, i]
                for n in k:N_b
                    maxv[:, n, i] .= gam_bound[1, i]
                    bprime[:, n, i] .= N_s
                    lamprime[:, n, i] .= N_l
                    debt_choice[:, n, i] .= Bline[N_s, 1]
                    maturity_choice[:, n, i] .= lam[N_l, 1]
                end
                break
            end
        end
    end
    
    @threads for l in 1:N_l
        for k in 1:N_b
            gam_prov = zeros(N_ex)
            gam_prov .= maxv[l, k, :]
            
            for i in 1:N_ex
                gam_prov[i] = max(gam_bound[i], gam_prov[i])
            end
            
            gam_prov .= gam_prov * InvTT
            gamma_pnew[l, k, :] .= gam_prov
        end
    end
end


###############################################################################

#------------------------------------------------------------------------------
#                               2.  PRICE SCHEDULE UPDATING 
#------------------------------------------------------------------------------

###############################################################################




# Update pricing schedule subroutine
function update_price!(q_new, q_old, gamma_p, gamma_a, gamma_ck, coll_points, coll_price, bounds, ss, price_index, Bline, lam, debt, weightsq, pointsq, bprime, lamprime, TTT2,
    index_s, index_ex1, index_ex2, s_prime, points_qy, points_qchi, points_qpi, points_y, points_chi, points_pi)

    # 1. Define variables
    N_ex, N_b, N_l, N_pq, N_s, N_price = size(q_new)  # Assume these are defined or passed as parameters
    N_p = size(gamma_p, 3)  # Extracting N_p from gamma_p's size
    phi_0, phi_1, kappa_0, kappa_1, sigma_chi = 0.1, 0.2, 0.3, 0.4, 0.5  # Example parameters, set as needed

    # Allocate storage for matrices
    TT_def = zeros(Float32, N_pq, 1)
    TT_nodef = zeros(Float32, N_pq, 1)
    TT_ck = zeros(Float32, N_pq, 1)
    sdf = zeros(Float32, N_pq, 1)
    safe = zeros(Float32, N_pq, 1)
    pay = zeros(Float32, N_pq, 1)
    payouts = zeros(Float32, N_pq, 1)
    price_tom = zeros(Float32, N_pq, N_l)
    price_prime = zeros(Float32, N_pq, 1)

    # Preallocate q_new
    q_new .= 0  # Reset q_new array

    # 2.1 Generate points for exogenous shocks
    y_y = zeros(Float32, 1, N_ex)
    chi_y = zeros(Float32, 1, N_ex)
    pi_y = zeros(Float32, 1, N_ex)

    for i in 1:N_ex
        y_y[1, i] = ss[1, 1] + coll_points[1, i]
        chi_y[1, i] = ss[2, 1] + coll_points[2, i]
        pi_y[1, i] = ss[3, 1] + coll_points[3, i]
    end

    # 2.2 Pricing schedule
    y_p = coll_price[1, :]
    chi_p = coll_price[2, :] .+ ss[2, 1]
    pi_p = coll_price[3, :] .+ ss[3, 1]

    # Parallel loop to compute q_new
    @threads for j in 1:N_price
        for o in 1:N_l
            for i in 1:N_s
                # Reset temporary arrays
                pay .= 0
                safe .= 0
                price_tom .= 0

                # Loop through possible states (or grid points)
                for k in 1:N_pq
                    # Compute sdf, Bprime, lamprime, and price_tom
                    sdf[k, 1] = exp(-(phi_0 + phi_1 * chi_p[j]) - 0.5 * (kappa_0 + kappa_1 * chi_p[j])^2 * sigma_chi^2 + sqrt(2) * (kappa_0 + kappa_1 * chi_p[j]) * sigma_chi * pointsq[2, k])

                    # Interpolate Bprime and lamprime (Assume interp_b and interp_lam functions exist)
                    Bprime_l = interp_b(points_y, points_chi, points_pi, index_ex1, collapse_ex, Bline, bprime, s_prime, j, k, index_s[i, 1], o)
                    Bprime_h = interp_b(points_y, points_chi, points_pi, index_ex1, collapse_ex, Bline, bprime, s_prime, j, k, index_s[i, 1] + 1, o)

                    xxx = Bprime_l * ((debt[index_s[i, 1] + 1, 1] - Bline[i, 1]) / (debt[index_s[i, 1] + 1, 1] - debt[index_s[i, 1], 1])) +
                          Bprime_h * ((Bline[i, 1] - debt[index_s[i, 1], 1]) / (debt[index_s[i, 1] + 1, 1] - debt[index_s[i, 1], 1]))

                    # Find the closest index to Bprime
                    index_bb = argmin(abs(Bline .- xxx[1]))

                    lamprime_l = interp_lam(points_y, points_chi, points_pi, index_ex1, collapse_ex, lam, lamprime, s_prime, j, k, index_s[i, 1], o)
                    lamprime_h = interp_lam(points_y, points_chi, points_pi, index_ex1, collapse_ex, lam, lamprime, s_prime, j, k, index_s[i, 1] + 1, o)

                    xxx = lamprime_l * ((debt[index_s[i, 1] + 1, 1] - Bline[i, 1]) / (debt[index_s[i, 1] + 1, 1] - debt[index_s[i, 1], 1])) +
                          lamprime_h * ((Bline[i, 1] - debt[index_s[i, 1], 1]) / (debt[index_s[i, 1] + 1, 1] - debt[index_s[i, 1], 1]))

                    # Find the closest index to lamprime
                    index_ll = argmin(abs(lam .- xxx[1]))

                    # Compute price_tom for all l (using interpolation)
                    for l in 1:N_l
                        price_tom[k, l] = max(interp_q(q_old, points_qy, points_qchi, points_qpi, index_ex2, collapse, index_bb, index_ll, s_prime, j, k, l), 0.0)
                    end

                    # Compute continuation fund components
                    cont_fund_u = 0
                    cont_fund_l = 0
                    cont_fund_uck = 0
                    cont_fund_lck = 0

                    for l in 1:N_ex
                        cont_fund_l += gamma_p[o, index_s[i, 1], l] * TTT2[j, k, l, 1]
                        cont_fund_u += gamma_p[o, index_s[i, 1] + 1, l] * TTT2[j, k, l, 1]
                        cont_fund_lck += gamma_ck[o, index_s[i, 1], l] * TTT2[j, k, l, 1]
                        cont_fund_uck += gamma_ck[o, index_s[i, 1] + 1, l] * TTT2[j, k, l, 1]
                    end

                    # Calculate continuation values
                    TT_nodef[k, 1] = cont_fund_l * ((debt[index_s[i, 1] + 1, 1] - Bline[i, 1]) / (debt[index_s[i, 1] + 1, 1] - debt[index_s[i, 1], 1])) +
                                      cont_fund_u * ((Bline[i, 1] - debt[index_s[i, 1], 1]) / (debt[index_s[i, 1] + 1, 1] - debt[index_s[i, 1], 1]))
                    TT_ck[k, 1] = cont_fund_lck * ((debt[index_s[i, 1] + 1, 1] - Bline[i, 1]) / (debt[index_s[i, 1] + 1, 1] - debt[index_s[i, 1], 1])) +
                                  cont_fund_uck * ((Bline[i, 1] - debt[index_s[i, 1], 1]) / (debt[index_s[i, 1] + 1, 1] - debt[index_s[i, 1], 1]))

                    cont_fund = 0
                    for l in 1:N_ex
                        cont_fund += gamma_a[1, l] * TTT2[j, k, l, 1]
                    end

                    TT_def[k, 1] = cont_fund
                    safe[k, 1] = TT_ck[k, 1] >= TT_def[k, 1] ? 1.0 : 0.0
                    pay[k, 1] = TT_nodef[k, 1] >= TT_def[k, 1] ? 1.0 : 0.0
                end

                # Calculate price prime and payouts
                for p in 1:N_l
                    price_prime[:, 1] .= price_tom[:, p]
                    payouts .= sdf .* (safe .* (lam[p, 1] + (1 - lam[p, 1]) .* price_prime) .+ (1 .- safe) .* pay .* (1 .- pi_p[j]) .* (lam[p, 1] + (1 - lam[p, 1]) .* price_prime))
                    cont_fund .= weightsq * payouts
                    q_new[i, o, p, j] .= (π^(-1.5)) * cont_fund[1, 1]
                end
            end
        end
    end

    # Adjust values of q_new across different price indices
    for i in 1:N_price
        for j in 1:N_s
            for m in 1:N_l
                for l in 1:N_l
                    if j > 1
                        q_new[j, m, l, i] = min(q_new[j, m, l, i], q_new[j - 1, m, l, i])
                    end
                end
            end
        end
    end

    return q_new
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

    # Allocate arrays
    # N_sl, N_su = zeros(Int, size(coll_points)...), zeros(Int, size(coll_points)...)
    # index_s = zeros(Int, size(coll_points)...)
    
    # New variables
    # index_ex1, index_ex2, index_ck = zeros(Int, size(coll_points)...), zeros(Int, size(coll_points)...), zeros(Int, size(coll_points)...)
    # s_prime = zeros(Float32, size(coll_points)...)
    
    # points_qy, points_qchi, points_qpi, points_y, points_chi, points_pi = zeros(Float32, size(coll_points)...), 
    # zeros(Float32, size(coll_points)...), zeros(Float32, size(coll_points)...), zeros(Float32, size(coll_points)...), zeros(Float32, size(coll_points)...), zeros(Float32, size(coll_points)...)
    
    # coll_transformed = zeros(Float32, size(coll_points)...)
    # coll_transformed1 = zeros(Float32, size(coll_price)...)
    ones_Np = ones(Float32, N_price)
    ones_Nc = ones(Float32, N_ex)
    
    # y_p = zeros(Float32, size(coll_price)...)
    # chi_p = zeros(Float32, size(coll_price)...)
    # pi_p = zeros(Float32, size(coll_price)...)
    
    # weightsq = zeros(Float32, size(coll_points)...)
    # pointsq = zeros(Float32, size(coll_points)...)
    
    x_prime = zeros(Float32, size(coll_points)...)
    y_prime = zeros(Float32, size(coll_points)...)
    
    # extra = zeros(Float32, size(coll_points, 1))
    # extra2 = zeros(Float32, size(coll_points, 1))

    # Allocate arrays
    # TO BE DELETED
    # y_p[1, :] .= coll_price[1, :]
    # chi_p[1, :] .= coll_price[2, :] .+ ss[2, 1]
    # pi_p[1, :] .= coll_price[3, :] .+ ss[3, 1]

    # extra[1, 1] = bounds[1, 1] + bounds[1, 2]
    # extra[2, 1] = bounds[2, 1] + bounds[2, 2]
    # extra[3, 1] = bounds[3, 1] + bounds[3, 2]
    
    # extra2[1, 1] .= bounds[1, 2] - bounds[1, 1]
    # extra2[2, 1] .= bounds[2, 2] - bounds[2, 1]
    # extra2[3, 1] .= bounds[3, 2] - bounds[3, 1]

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
