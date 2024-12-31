using CSV
using XLSX
using DataFrames
using HTTP
using NLsolve
using Statistics
using Plots

# ==========================================================================
#         Part A - Computing Data Average Excess Returns in the Data
# ==========================================================================

# Load Italian yields data
url = "https://raw.githubusercontent.com/maestri-and/Bocola_Dovis-2019/refs/heads/main/data/ItalianYields.csv"
# Download and read the CSV file
response = HTTP.get(url)
data = CSV.File(IOBuffer(response.body))
# Convert to a DataFrame
yieldcurve = DataFrame(data)

#Load german yield data
url = "https://raw.githubusercontent.com/maestri-and/Bocola_Dovis-2019/refs/heads/main/data/German3moYields.csv"
# Download and read the CSV file
response = HTTP.get(url)
data2 = CSV.File(IOBuffer(response.body))
# Convert to a DataFrame
german_3mo_yields = DataFrame(data2)

# ==========================================================================
#    Part A.1 - Compute prices and yields of ZCB using Italian yields data
# ==========================================================================
# Extract relevant variables
Y = Matrix(yieldcurve)  # Convert DataFrame to a Matrix for easier manipulation
time = range(1996.5, stop=2016.5, step=1/12)
T = size(yieldcurve, 1)
aguess = [0.0, 0.0, 0.0, 0.0]  # Initial guess for the solver

# Pre-allocate matrix to store results
A3 = zeros(Float64, T, 4)

# Define the polynomial function (equivalent to @poly3 in MATLAB)
function poly3(a)
    global y
    f = zeros(4)  # Initialize a 4-element vector
    f[1] = a[1] + a[2]*1 + a[3]*1^2 + a[4]*1^3 - y[1]
    f[2] = a[1] + a[2]*3 + a[3]*3^2 + a[4]*3^3 - y[2]
    f[3] = a[1] + a[2]*5 + a[3]*5^2 + a[4]*5^3 - y[3]
    f[4] = a[1] + a[2]*10 + a[3]*10^2 + a[4]*10^3 - y[4]
    return f
end

# Loop through each row of Y
for t in 1:T
    global y = [Y[t, 1], Y[t, 3], Y[t, 5], Y[t, 10]]  # Select specific columns
    result = nlsolve(poly3, aguess)  # Solve the nonlinear system
    A3[t, :] = result.zero  # Store the solution
    aguess = result.zero  # Update the guess for the next iteration
end

# Initialize parameters
N_zcb = 80  # Number of zero-coupon bonds (ZCBs)
prices_zcb = zeros(T, N_zcb)  # Matrix to store ZCB prices
yields_zcb = zeros(T, N_zcb)  # Matrix to store ZCB yields

# Compute ZCB prices and yields
for t in 1:T
    for n in 1:N_zcb
        # Calculate yield (r) using the coefficients in A3
        r = (A3[t, 1] + A3[t, 2] * (n / 4) + A3[t, 3] * (n / 4)^2 + A3[t, 4] * (n / 4)^3) / 100
        # Calculate price (p) of the zero-coupon bond
        p = 1 / (1 + r)^(n / 4)
        # Store results in matrices
        prices_zcb[t, n] = p
        yields_zcb[t, n] = r
    end
end

# Find the starting and ending indices for a specific time range
time_array = collect(time)  # Convert time range to an array if necessary
starting_index = findmin(abs.(time_array .- 2000))[2]  # Closest index to 2000
ending_index = findmin(abs.(time_array .- 2013.9))[2]  # Closest index to 2013.9


# ==========================================================================
#           Part A.2 - Construct prices and yields of portfolio
# ==========================================================================

# Select the relevant time range and corresponding data
time_ita_mon = time_array[starting_index:ending_index]
prices_zcb = prices_zcb[starting_index:ending_index, :]
yields_zcb = yields_zcb[starting_index:ending_index, :]

# Define `lam` (scaling factor for the portfolio construction)
lam = vcat(range(0.5, stop=8, length=30), [5.0])  # Combine linspace and an additional value
lam = (4 .* lam).^(-1)  # Scale lambda values

# Preallocate matrices for portfolio prices and yields
price_portfolio = zeros(size(prices_zcb, 1), 31)  # Same row count, 30 columns
yields_portfolio = zeros(size(prices_zcb, 1), 31)  # Same row count, 30 columns

# Loop to calculate portfolio prices and yields
for j in 1:31  # Loop over 30 values of lambda
    lambda = lam[j]

    # Create `index` and compute `weights`
    index = collect(1:N_zcb-1)  # Create index vector (1 to N_zcb-1)
    weights = (1 - lambda) .^ (index .- 1)  # Element-wise power calculation

    # Compute portfolio price
    price_portfolio[:, j] = lambda * (
        prices_zcb[:, 1:N_zcb-1] * weights .+  # Element-wise multiplication
        ((1 - lambda)^(N_zcb-1) / lambda) .* prices_zcb[:, end]  # Broadcast scalar to vector
    )

    # Compute portfolio yield
    yields_portfolio[:, j] = lambda * (1 .- price_portfolio[:, j]) ./ price_portfolio[:, j]
    
    # Print `weights` for debugging
    println("Lambda (j = $j): $lambda")
    println("Weights: ", weights)
end

# ==========================================================================
#     Part A.3 - Compute average holding period returns for portfolios
# ==========================================================================

T = size(price_portfolio, 1)  # Number of rows in price_portfolio
RR = zeros(T - 1, 30)  # Initialize RR matrix for 30 portfolios

# Loop over portfolios
for k in 1:30
    lambda = lam[k]  # Extract lambda value for portfolio k
    
    # Compute holding period returns
    RR[:, k] = (lambda .+ (1 .- lambda) .* price_portfolio[2:end, k]) ./ price_portfolio[1:end-1, k]
end

# Find indices for pre-crisis and crisis periods
precrisis_l = argmin(abs.(time_ita_mon .- 2000))
precrisis_u = argmin(abs.(time_ita_mon .- 2007.75))
crisis_l = argmin(abs.(time_ita_mon .- 2008.00))
crisis_u = argmin(abs.(time_ita_mon .- 2012.75))

german = Matrix(german_3mo_yields)  # Convert to a matrix

time_g = german[:, 1]  # Extract time column
precrisis_lg = argmin(abs.(time_g .- 2000))
precrisis_ug = argmin(abs.(time_g .- 2007.75))
crisis_lg = argmin(abs.(time_g .- 2008.00))
crisis_ug = argmin(abs.(time_g .- 2012.75))

# Compute excess holding period returns
eer_precrisis = (mean(RR[precrisis_l:precrisis_u, :], dims=1) .- 1) .* 400 .- mean(german[precrisis_lg:precrisis_ug, 2]) .* 100
eer_crisis = (mean(RR[crisis_l:crisis_u, :], dims=1) .- 1) .* 400 .- mean(german[crisis_lg:crisis_ug, 2]) .* 100

# ==========================================================================
#        Part B - Representing the Average Excess Returns Graph
# ==========================================================================

# Define colors for the plot
colorb = RGB(0, 0.4470, 0.7410)  # Blue (similar to MATLAB)
colorr = RGB(0.82, 0.20, 0.2)    # Red (similar to MATLAB)

# Calculate maturity for x-axis
maturity = 1 ./ (4 .* lam)
# Slice maturity to match the size of eer_crisis and eer_precrisis
maturity = maturity[1:30]

# Create the plot
plot(
    maturity_30, eer_precrisis',
    label="Data, pre-crisis (2000-2007)",
    linewidth=3,
    linestyle=:dash,
    color=colorb,
    grid=true,  # Enable grid
    gridstyle=:dash,  # Set grid style to dashed
    gridcolor=:black,  # Set grid color to black
    gridalpha=0.5,  # Set grid transparency
    size=(400, 400)  # Set figure size
)

plot!(
    maturity_30, eer_crisis',
    label="Data, crisis (2008-2012)",
    linewidth=3,
    linestyle=:solid,
    color=colorb,
    legend=:bottomright,  # Position the legend in the bottom-left corner
)

# Customize plot appearance
xlabel!("Maturity")
ylabel!("Average excess returns")
xlims!(0, maximum(maturity_30))
ylims!(0, 3.5)  # Adjust as necessary for the data range
title!("Panel B. Average excess returns")