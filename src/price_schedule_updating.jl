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
    y_y = zeros(Float32, N_ex)
    chi_y = zeros(Float32, N_ex)
    pi_y = zeros(Float32, N_ex)

    y_y = ss[1,1] .+ coll_points[1, :]
    chi_y = ss[2,1] .+ coll_points[2, :]
    pi_y = ss[3,1] .+ coll_points[3, :]

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
                        cont_fund += gamma_a[l] * TTT2[j, k, l, 1]
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
