using XLSX
using HTTP
using Plots
using DataFrames
using CSV

# Step 1: Define the Remote URL and Local Path
remote_url = "https://raw.githubusercontent.com/maestri-and/Bocola_Dovis-2019/refs/heads/main/data/Data.xlsx"  # Replace with your raw Git URL
local_path = "Data.xlsx"  # Local file path to save the downloaded file

# Step 2: Download the File
HTTP.open(:GET, remote_url) do file
    open(local_path, "w") do local_file
        write(local_file, read(file))
    end
end

# Step 3: Load the Excel File
xf = XLSX.readxlsx(local_path)
println("Sheetnames: ", XLSX.sheetnames(xf))

#############################################################################
                              #Panel A
#############################################################################

# Import the relevant sheet into a data frame
sh1 = xf["GDP"] 
@show sh1["A1:B2"]
df = DataFrame(XLSX.readtable("/Users/alibenra/Desktop/Julia Files/Datanew.xlsx", "GDP"))
CSV.write("output.csv", df)

# Extract the year and quarter from the Date column (e.g., "Q1-2000")
split_dates = split.(df.Date, "-")  # Split the Date string by "-"
# Extract Year (second part of the split)
df.Year = Int64.(map(x -> parse(Int, x[2]), split_dates))
# Extract Quarter (first part of the split)
df.Quarter = map(x -> x[1], split_dates)
# Convert the Quarter to a decimal year format (e.g., Q1 = 0.0, Q2 = 0.25, Q3 = 0.5, Q4 = 0.75)
df.Time = [df.Year[i] + (parse(Int, df.Quarter[i][2]) - 1) * 0.25 for i in 1:nrow(df)]  # Generate Time column directly  # Initialize an empty array for the time vector
# Extract GDP data and corresponding Date
gdp_data = df[:, [:Time, :GDP]]
# Filter the data to start at 2000
filtered_gdp_data = gdp_data[(df.Time .>= 2000.0) .& (df.Time .<= 2012.25), :]

# Take the log of GDP for the selected data
log_gdp = log.(filtered_gdp_data.GDP)
norm =log_gdp[1]
log_gdp_norm = log_gdp .- log_gdp[1]

# Plot Panel A: Output (log of GDP)
xticks = 2000:2:2012  # x-axis labels from 2000 to 2012, with a 2-year gap
xtick_labels = string.(xticks) 
p1 = plot(filtered_gdp_data.Time, log_gdp_norm, ylabel="Log of GDP normalized", label="Log of GDP", title="Panel A: Output", xticks=(xticks, xtick_labels))
savefig("panel_a_output.png")

#############################################################################
                              #Panel B
#############################################################################
# Import the relevant sheet into a data frame
df2 = DataFrame(XLSX.readtable("/Users/alibenra/Desktop/Julia Files/Datanew.xlsx", "Main Series"))
CSV.write("debt2output.csv", df2)
# Extract the year and quarter from the Date column (e.g., "Q1-2000")
split_dates2 = split.(df2.Date, "Q")  # Split the Date string by "-"
# Extract Year (second part of the split)
df2.Year = Int64.(map(x -> parse(Int, x[1]), split_dates2))
# Extract Quarter (first part of the split)
df2.Quarter = map(x -> x[2], split_dates2)
# Convert the Quarter to a decimal year format (e.g., Q1 = 0.0, Q2 = 0.25, Q3 = 0.5, Q4 = 0.75)
df2.Time = [df2.Year[i] + (parse(Int, df2.Quarter[i]) - 1) * 0.25 for i in 1:nrow(df2)]  # Create the Time column
df2 = df2[(df2.Time .>= 2000.0) .& (df2.Time .<= 2012.25), :]

rename!(df2, "Debt maturity" => :Debt_maturity)

# Computing new issuances
# Step 1: Read the "Redemption profile Italian debt" sheet into a DataFrame
df_redemption = DataFrame(XLSX.readtable("/Users/alibenra/Desktop/Julia Files/Datanew.xlsx", "Redemption profile Italian debt"))
# Convert the redemption profile into a matrix and exclude the Date column
S = Matrix(df_redemption[:, 2:end])  # Redemption profile matrix (debt redemptions)

# Define the time vector
time = 2009.0:1:2015.0  # Time vector from 2009.5 to 2015.5

# Step 2: Extract Every 4th Row for Redemption Profile
# Initialize an empty redemption matrix
redemption = Matrix{Float64}(undef, 0, size(S, 2))  # Start with 0 rows and N columns
counter = 3  # Start counter at 3 (as in Matlab)

# Loop through rows of the redemption profile to extract every 4th row
for j in 1:size(S, 1)
    counter += 1
    if counter == 4
        redemption = vcat(redemption, S[j, :]')  # Append the row to redemption
        counter = 0
    end
end

# Step 3: Calculate New Issuances
# Compute the differences between consecutive rows of the redemption profile
new_issuances = hcat(
    redemption[2:end, 1:end-1] .- redemption[1:end-1, 2:end],  # Differences for all columns except the last
    redemption[2:end, end]  # Add the last column
)

# Compute the Weighted Average
weights = range(1, stop=size(redemption, 2), length=size(redemption, 2))  # Weights from 1 to N
new_issuances_weighted = sum(new_issuances .* weights', dims=2) ./ sum(new_issuances, dims=2)
new_issuances_time = time[1:end]

# Plotting
# Step 1: Define Debt Maturity (Stock) Data
x1 = df2.Time  # Time for Debt Maturity
y1 = df2.Debt_maturity  # Debt Maturity values
# Step 2: Define New Issuances Data
x2 = new_issuances_time[(new_issuances_time .<= 2012.5)]  # Filter x2 to only include time <= 2012.5
y2 = new_issuances_weighted[1:length(x2)]  # Ensure y2 matches the filtered x2
# Step 3: Create the Dual-Axis Plot
p2 = plot(
    x1, 
    y1, 
    label="Stock", 
    xlabel="Year", 
    yaxis="Debt Maturity (Stock)", 
    title="Panel B: Debt Maturity", 
    color=:blue, 
    legend=:topleft,
    xlims=(2000,2013),
    xticks=[2000, 2002, 2004, 2006, 2008, 2010, 2012],
    ylims=(6.55, 7.2),  # Set y-axis range for Debt Maturity
    yticks=[6.6, 6.8, 7.0, 7.2]
)
# Overlay New Issuances (Scatter Points) with a Second Y-Axis
plot!(
    twinx(),  # Activate second y-axis
    x2, 
    y2, 
    seriestype=:scatter, 
    label="New Issuances (right scale)", 
    yaxis="New Issuances", 
    color=:red, 
    markershape=:circle,
    ylims=(5, 9)
)
savefig("panel_b_maturity.png")


#############################################################################
                              #Panel C
#############################################################################

rename!(df2, "Debt to output ratio" => :Debt_to_output_ratio)

# Plotting
p3 = plot(df2.Time, df2.Debt_to_output_ratio, label="Debt to output ratio", title="Panel C: Debt to output ratio", xticks=(xticks, xtick_labels), ylims=(80, 105))  
savefig("panel_c_debt_ratio.png")

#############################################################################
                              #Panel D
#############################################################################

# Plotting
p4 = plot(df2.Time, df2.Spread, label="Spreads", title="Panel D: ITA-GER spreads", xticks=(xticks, xtick_labels), ylims=(-0.5, 5), yticks=[0, 1, 2,3,4,5])  
savefig("panel_c_debt_ratio.png")