###############################################################################
########################## AUXILIARY_FUNCTIONS.JL #############################

############ This script defines additional functions used to solve ###########
################## the benchmark Bocola & Dovis [2019] model ##################

###############################################################################

#------------------------------------------------------------------------------
#              1. Importing libraries and pre-allocated parameters
#------------------------------------------------------------------------------


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
# - Matrix Inversion: Use `inv(A)` from the `LinearAlgebra` package.
# - Binary Search: Use `searchsorted(xa, x)` from the Julia standard library.
# - Recurrence Relations and Matrix Operations: Use Julia's built-in matrix and array operations
#   (e.g., `*`, `.*`, and `transpose`) for efficient computation.

###############################################################################

# ###############################################################################

# #------------------------------------------------------------------------------
# #                               3a.  GAUSS MATRIX INVERSION 
# #------------------------------------------------------------------------------

# ###############################################################################





# function gauss!(a::Matrix{Float32}, n::Int)
#     # Invert matrix by Gauss method
#     b = copy(a)  # Make a copy of the input matrix
#     ipvt = 1:n   # Create a vector of pivot indices
#     for k in 1:n
#         # Find the pivot element (index of max absolute value in column k)
#         imax = argmax(abs(b[k:n, k]))[1] + k - 1
#         m = imax
#         if m != k
#             # Swap rows k and m
#             ipvt[k], ipvt[m] = ipvt[m], ipvt[k]
#             b[[k, m], :] .= b[[m, k], :]
#         end
#         d = 1 / b[k, k]
#         temp = b[:, k]
#         for j in 1:n
#             c = b[k, j] * d
#             b[:, j] .-= temp * c
#             b[k, j] = c
#         end
#         b[:, k] .= temp * -d
#         b[k, k] = d
#     end
#     a[:, ipvt] .= b
# end



# ###############################################################################

# #------------------------------------------------------------------------------
# #                               3b.  TGEN - GENERATE T MATRIX
# #------------------------------------------------------------------------------

# ###############################################################################



# function Tgen!(T_final::Matrix{Float32}, X::Vector{Float32}, N::Int, state_ex::Int, N_ex::Int)
#     # Generate T matrix
#     T = ones(Float32, 1, N, state_ex)  # Initialize T as a 3D matrix of size (1, N, state_ex)
#     T_prod_trans = ones(Float32, 1, N_ex)  # Initialize the T_prod_trans vector

#     # Step 1: Populate T with values based on X
#     for i in 1:state_ex
#         T[1, 2, i] = X[i]
#     end

#     # Step 2: Generate the recurrence relation for T
#     for i in 3:N
#         for l in 1:state_ex
#             T[1, i, l] = 2 * X[l] * T[1, i-1, l] - T[1, i-2, l]
#         end
#     end

#     # Step 3: Compute T_prod_trans based on T
#     k = 0
#     for i in 1:N
#         for l in 1:N
#             for ii in 1:N
#                 k += 1
#                 T_prod_trans[1, k] = T[1, i, 1] * T[1, l, 2] * T[1, ii, 3]
#             end
#         end
#     end

#     # Final transpose to get the result
#     T_final .= transpose(T_prod_trans)
# end


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
