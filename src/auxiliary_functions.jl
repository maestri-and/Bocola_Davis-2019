###############################################################################
########################## AUXILIARY_FUNCTIONS.JL #############################

############ This script defines additional functions used to solve ###########
################## the benchmark Bocola & Dovis [2019] model ##################

###############################################################################

#------------------------------------------------------------------------------
#              1. Importing libraries and pre-allocated parameters
#------------------------------------------------------------------------------

using LinearAlgebra
using Interpolations
using LoopVectorization

###############################################################################

# ###############################################################################

# #------------------------------------------------------------------------------
# #                               1.  T_NOSMOLYAK 
# #------------------------------------------------------------------------------

# ###############################################################################

function T_nosmolyak(y, N, d)
    # Transpose y to match the MATLAB behavior
    y = y'

    # Initialize T as a 3D array: (size(y, 1), N, d)
    T = ones(size(y, 1), N, d)

    # Compute the Chebyshev polynomials for each dimension
    for dind in 1:d
        T[:, 2, dind] .= y[:, dind]
    end

    # Precompute 2 * y for efficiency
    twoz = 2 .* y

    # Loop to compute the recursive Chebyshev polynomials
    for oind in 3:N
        for dind in 1:d
            T[:, oind, dind] .= twoz[:, dind] .* T[:, oind-1, dind] .- T[:, oind-2, dind]
        end
    end

    # Initialize the product matrix Tprod
    Tprod = ones(size(y, 1), N^d)

    # Compute the product across all combinations of the dimensions
    index = 1  # This will keep track of the column index for Tprod
    for i1 in 1:N
        for i2 in 1:N
            for i3 in 1:N
                # For each combination of indices (i1, i2, ..., id), compute the product of T polynomials
                Tprod[:, index] .= T[:, i1, 1] .* T[:, i2, 2] .* T[:, i3, 3]
                index += 1
            end
        end
    end
    
    # Now Tprod should have size (size(y, 1), N^d), and we need to transpose it
    T = Tprod'

    return T
end


###############################################################################

################################## IMPORTANT !!! ##############################

# For the functions in 3:
# 
# - Binary Search: Use `searchsorted(xa, x)` from the Julia standard library.
# - Recurrence Relations and Matrix Operations: Use Julia's built-in matrix and array operations
#   (e.g., `*`, `.*`, and `transpose`) for efficient computation.


# ###############################################################################

# #------------------------------------------------------------------------------
# #                               3b.  TGEN - GENERATE T MATRIX
# #------------------------------------------------------------------------------

# ###############################################################################

####### VECTORISED VERSION

function Tgen!(T_final, X, N, state_ex, N_ex)
    # Initialize T and T_prod_trans
    T = ones(1, N, state_ex)  # Allocate T within the function
    T_prod_trans = ones(1, N_ex)

    # Populate T based on X
    for i in 1:state_ex
        T[1, 2, i] = X[1, i]
    end

    for i in 3:N
        for l in 1:state_ex
            T[1, i, l] = 2 * X[1, l] * T[1, i - 1, l] - T[1, i - 2, l]
        end
    end

    # Compute T_prod_trans
    k = 0
    for i in 1:N, l in 1:N, ii in 1:N
        k += 1
        T_prod_trans[1, k] = T[1, i, 1] * T[1, l, 2] * T[1, ii, 3]
    end

    # Transpose the result into T_final
    T_final .= transpose(T_prod_trans)
end



# ###############################################################################

# #------------------------------------------------------------------------------
# #                               3c.  DEFINE BISECT - BINARY SEARCH
# #------------------------------------------------------------------------------

# ###############################################################################


# function bisect!(xa::Vector{Float32}, x::Float32)
#     # Perform binary search for x in the vector xa
#     n = length(xa)
#     l, u = 1, n
#     i = 0  # Default index if no match is found

#     if x > xa[u]
#         i = n
#     elseif x < xa[l]
#         i = 0
#     else
#         for t in 1:n
#             m = Int((l + u) / 2)
#             if x > xa[m]
#                 l = m
#             else
#                 u = m
#             end
#             if (l + u) ÷ 2 == l
#                 i = l
#                 break
#             end
#         end
#     end
#     return i
# end






###############################################################################

################################## IMPORTANT !!! ##############################

# For the functions in 5:
# 
# - They can be redone using Interpolations.jl - SEE NEXT

###############################################################################

# ###############################################################################

# #------------------------------------------------------------------------------
# #                               5.  INTERPOLATION - MANUAL
# #------------------------------------------------------------------------------

# ###############################################################################



# #------------------------------------------------------------------------------
# #                               A.  INT_3D
# #------------------------------------------------------------------------------


# function int_3D(x::Vector{Float32}, c::Vector{Float32}, y::Vector{Float32})
#     # x contains the 6 coordinate values (x0, x1, y0, y1, z0, z1)
#     # c contains the 8 values at the corners of the cube (c000, c100, c010, ..., c111)
#     # y is the target point at which to interpolate (y1, y2, y3)

#     xd = (y[1] - x[1]) / (x[2] - x[1])  # interpolation coefficient along x-axis
#     yd = (y[2] - x[3]) / (x[4] - x[3])  # interpolation coefficient along y-axis
#     zd = (y[3] - x[5]) / (x[6] - x[5])  # interpolation coefficient along z-axis

#     # Interpolation in the x-direction
#     c00 = c[1] * (1 - xd) + c[2] * xd
#     c01 = c[4] * (1 - xd) + c[6] * xd
#     c10 = c[3] * (1 - xd) + c[5] * xd
#     c11 = c[7] * (1 - xd) + c[8] * xd

#     # Interpolation in the y-direction
#     c0  = c00 * (1 - yd) + c10 * yd
#     c1  = c01 * (1 - yd) + c11 * yd

#     # Final interpolation in the z-direction
#     return c0 * (1 - zd) + c1 * zd
# end

# # #------------------------------------------------------------------------------
# # #                               B.  INTERP_B
# # #------------------------------------------------------------------------------

# function interp_b(points_y, points_chi, points_pi, index_ex1, collapse_ex, Bline, bprime, s_prime, j, k, i, o)
#     # Initialize the coordinates for the 6 surrounding points (x0, x1, y0, y1, z0, z1)
#     x = Float32[
#         points_y[index_ex1[1, j, k], 1], points_y[index_ex1[1, j, k] + 1, 1],
#         points_chi[index_ex1[2, j, k], 1], points_chi[index_ex1[2, j, k] + 1, 1],
#         points_pi[index_ex1[3, j, k], 1], points_pi[index_ex1[3, j, k] + 1, 1]
#     ]

#     # Initialize the 8 function values at the corners of the cube
#     c = Float32[
#         Bline[bprime[o, i, collapse_ex[index_ex1[1, j, k], index_ex1[2, j, k], index_ex1[3, j, k]]], 1],
#         Bline[bprime[o, i, collapse_ex[index_ex1[1, j, k] + 1, index_ex1[2, j, k], index_ex1[3, j, k]]], 1],
#         Bline[bprime[o, i, collapse_ex[index_ex1[1, j, k], index_ex1[2, j, k] + 1, index_ex1[3, j, k]]], 1],
#         Bline[bprime[o, i, collapse_ex[index_ex1[1, j, k], index_ex1[2, j, k], index_ex1[3, j, k] + 1]], 1],
#         Bline[bprime[o, i, collapse_ex[index_ex1[1, j, k] + 1, index_ex1[2, j, k] + 1, index_ex1[3, j, k]]], 1],
#         Bline[bprime[o, i, collapse_ex[index_ex1[1, j, k] + 1, index_ex1[2, j, k], index_ex1[3, j, k] + 1]], 1],
#         Bline[bprime[o, i, collapse_ex[index_ex1[1, j, k], index_ex1[2, j, k] + 1, index_ex1[3, j, k] + 1]], 1],
#         Bline[bprime[o, i, collapse_ex[index_ex1[1, j, k] + 1, index_ex1[2, j, k] + 1, index_ex1[3, j, k] + 1]], 1]
#     ]

#     # Perform the trilinear interpolation
#     return int_3D(x, c, s_prime[:, j, k])
# end

# # #------------------------------------------------------------------------------
# # #                               C.  INTERP_LAM
# # #------------------------------------------------------------------------------


# function interp_lam(points_y, points_chi, points_pi, index_ex1, collapse_ex, lam, lamprime, s_prime, j, k, i, o)
#     # Initialize the coordinates for the 6 surrounding points (x0, x1, y0, y1, z0, z1)
#     x = Float32[
#         points_y[index_ex1[1, j, k], 1], points_y[index_ex1[1, j, k] + 1, 1],
#         points_chi[index_ex1[2, j, k], 1], points_chi[index_ex1[2, j, k] + 1, 1],
#         points_pi[index_ex1[3, j, k], 1], points_pi[index_ex1[3, j, k] + 1, 1]
#     ]

#     # Initialize the 8 function values at the corners of the cube
#     c = Float32[
#         lam[lamprime[o, i, collapse_ex[index_ex1[1, j, k], index_ex1[2, j, k], index_ex1[3, j, k]]], 1],
#         lam[lamprime[o, i, collapse_ex[index_ex1[1, j, k] + 1, index_ex1[2, j, k], index_ex1[3, j, k]]], 1],
#         lam[lamprime[o, i, collapse_ex[index_ex1[1, j, k], index_ex1[2, j, k] + 1, index_ex1[3, j, k]]], 1],
#         lam[lamprime[o, i, collapse_ex[index_ex1[1, j, k], index_ex1[2, j, k], index_ex1[3, j, k] + 1]], 1],
#         lam[lamprime[o, i, collapse_ex[index_ex1[1, j, k] + 1, index_ex1[2, j, k] + 1, index_ex1[3, j, k]]], 1],
#         lam[lamprime[o, i, collapse_ex[index_ex1[1, j, k] + 1, index_ex1[2, j, k], index_ex1[3, j, k] + 1]], 1],
#         lam[lamprime[o, i, collapse_ex[index_ex1[1, j, k], index_ex1[2, j, k] + 1, index_ex1[3, j, k] + 1]], 1],
#         lam[lamprime[o, i, collapse_ex[index_ex1[1, j, k] + 1, index_ex1[2, j, k] + 1, index_ex1[3, j, k] + 1]], 1]
#     ]

#     # Perform the trilinear interpolation
#     return int_3D(x, c, s_prime[:, j, k])
# end

# # #------------------------------------------------------------------------------
# # #                               D.  INTERP_Q
# # #------------------------------------------------------------------------------

# function interp_q(q_old, points_qy, points_qchi, points_qpi, index_ex2, collapse, index_bb, index_ll, s_prime, j, k, o)
#     # Initialize the coordinates for the 6 surrounding points (x0, x1, y0, y1, z0, z1)
#     x = Float32[
#         points_qy[index_ex2[1, j, k], 1], points_qy[index_ex2[1, j, k] + 1, 1],
#         points_qchi[index_ex2[2, j, k], 1], points_qchi[index_ex2[2, j, k] + 1, 1],
#         points_qpi[index_ex2[3, j, k], 1], points_qpi[index_ex2[3, j, k] + 1, 1]
#     ]

#     # Initialize the 8 function values at the corners of the cube
#     c = Float32[
#         q_old[index_bb[o, collapse[index_ex2[1, j, k], index_ex2[2, j, k], index_ex2[3, j, k]]], index_ll[i], 1],
#         q_old[index_bb[o, collapse[index_ex2[1, j, k] + 1, index_ex2[2, j, k], index_ex2[3, j, k]]], index_ll[i], 1],
#         q_old[index_bb[o, collapse[index_ex2[1, j, k], index_ex2[2, j, k] + 1, index_ex2[3, j, k]]], index_ll[i], 1],
#         q_old[index_bb[o, collapse[index_ex2[1, j, k], index_ex2[2, j, k], index_ex2[3, j, k] + 1]], index_ll[i], 1],
#         q_old[index_bb[o, collapse[index_ex2[1, j, k] + 1, index_ex2[2, j, k] + 1, index_ex2[3, j, k]]], index_ll[i], 1],
#         q_old[index_bb[o, collapse[index_ex2[1, j, k] + 1, index_ex2[2, j, k], index_ex2[3, j, k] + 1]], index_ll[i], 1],
#         q_old[index_bb[o, collapse[index_ex2[1, j, k], index_ex2[2, j, k] + 1, index_ex2[3, j, k] + 1]], index_ll[i], 1],
#         q_old[index_bb[o, collapse[index_ex2[1, j, k] + 1, index_ex2[2, j, k] + 1, index_ex2[3, j, k] + 1]], index_ll[i], 1]
#     ]

#     # Perform the trilinear interpolation
#     return int_3D(x, c, s_prime[:, j, k])
# end


# ###############################################################################

# #------------------------------------------------------------------------------
# #                   5.  INTERPOLATIONS using Interpolations.jl
# #------------------------------------------------------------------------------

# ###############################################################################

# Helper function to create an Interpolation object
function create_interpolation(points_y, points_chi, points_pi, data_values)
    # Assume that points_y, points_chi, points_pi are 1D arrays of grid points
    # and data_values is a 3D matrix corresponding to the function values on the grid.

    # Create a 3D grid interpolation object using linear interpolation
    return interpolate(data_values, Gridded(Linear()), (points_y, points_chi, points_pi))
end

# Function for Bline interpolation
function interp_b(points_y, points_chi, points_pi, index_ex1, collapse_ex, Bline, bprime, s_prime, j, k, i, o)
    # Extract 3D grid points and values
    x = [
        points_y[index_ex1[1, j, k], 1], points_y[index_ex1[1, j, k] + 1, 1],
        points_chi[index_ex1[2, j, k], 1], points_chi[index_ex1[2, j, k] + 1, 1],
        points_pi[index_ex1[3, j, k], 1], points_pi[index_ex1[3, j, k] + 1, 1]
    ]

    # Collapse the indices and extract the corresponding values
    collapse_idx = collapse_ex[index_ex1[1, j, k], index_ex1[2, j, k], index_ex1[3, j, k]]
    bprime_values = bprime[o, i, collapse_idx]

    # Use Interpolations.jl to create the interpolation object
    interp = create_interpolation(points_y, points_chi, points_pi, Bline)

    # Perform the interpolation using the created object
    return interpolate(interp, x...)
end

# Function for Lam interpolation
function interp_lam(points_y, points_chi, points_pi, index_ex1, collapse_ex, lam, lamprime, s_prime, j, k, i, o)
    # Extract 3D grid points and values
    x = [
        points_y[index_ex1[1, j, k], 1], points_y[index_ex1[1, j, k] + 1, 1],
        points_chi[index_ex1[2, j, k], 1], points_chi[index_ex1[2, j, k] + 1, 1],
        points_pi[index_ex1[3, j, k], 1], points_pi[index_ex1[3, j, k] + 1, 1]
    ]

    # Collapse the indices and extract the corresponding values
    collapse_idx = collapse_ex[index_ex1[1, j, k], index_ex1[2, j, k], index_ex1[3, j, k]]
    lamprime_values = lamprime[o, i, collapse_idx]

    # Use Interpolations.jl to create the interpolation object
    interp = create_interpolation(points_y, points_chi, points_pi, lam)

    # Perform the interpolation using the created object
    return interpolate(interp, x...)
end

# Function for Q interpolation
function interp_q(q_old, points_qy, points_qchi, points_qpi, index_ex2, collapse, index_bb, index_ll, s_prime, j, k, o)
    # Extract 3D grid points and values
    x = [
        points_qy[index_ex2[1, j, k], 1], points_qy[index_ex2[1, j, k] + 1, 1],
        points_qchi[index_ex2[2, j, k], 1], points_qchi[index_ex2[2, j, k] + 1, 1],
        points_qpi[index_ex2[3, j, k], 1], points_qpi[index_ex2[3, j, k] + 1, 1]
    ]

    # Collapse the indices and extract the corresponding values
    collapse_idx = collapse[index_ex2[1, j, k], index_ex2[2, j, k], index_ex2[3, j, k]]
    q_old_values = q_old[index_bb[1], index_ll[1], o, collapse_idx]

    # Use Interpolations.jl to create the interpolation object
    interp = create_interpolation(points_qy, points_qchi, points_qpi, q_old)

    # Perform the interpolation using the created object
    return interpolate(interp, x...)
end
