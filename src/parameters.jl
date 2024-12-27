###############################################################################
############################### PARAMETERS.JL #################################

############ This script defines and assigns the parameters of ################
################ the benchmark Bocola & Dovis (2019) model ####################

###############################################################################

#Defining parameters to solve the model 

struct ModelParameters
    # Government's decision problem parameters
    sigma::Float64
    psi::Float64
    tau::Float64
    G::Float64
    mu_y::Float64
    rho_y::Float64
    sigma_y::Float64
    sigma_ychi::Float64
    beta::Float64
    d0::Float64
    d1::Float64
    alpha::Float64
    dbar::Float64
    pi_bar::Float64
    sigma_pi::Float64

    # Stochastic discount factor parameters
    phi0::Float64
    phi1::Float64
    kappa0::Float64
    kappa1::Float64
    rho_chi::Float64
end


# Setting parameters to be the same as in the paper (Table 1), after calibration
par_model = ModelParameters(
    2.000,  # σ
    0.050,  # ψ
    0.410,  # τ
    0.680,  # G
    0.892,  # μy
    0.970,  # ρy
    0.008,  # σy
    -0.002, # σyχ
    0.980,  # β
    0.058,  # d₀
    0.092,  # d₁
    0.400,  # α
    6.810,  # d̄
    -6.500, # π̄
    1.650,  # σπ
    0.005,  # φ₀
    0.002,  # φ₁
    0.161,  # κ₀
    0.374,  # κ₁
    0.513   # ρχ
)