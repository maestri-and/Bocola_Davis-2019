###############################################################################
######################### BENCHMARK_MODEL_SOLUTION.JL #########################

######### This script solves the benchmark Bocola & Dovis (2019) model ########

###############################################################################

#################### 0. IMPORTING LIBRARIES AND SCRIPTS #######################

using Statistics

# Recall other scripts
include("parameters.jl")

###############################################################################

################ 1. PREPARE GRIDS AND INITIAL GUESS FOR VFI ###################
