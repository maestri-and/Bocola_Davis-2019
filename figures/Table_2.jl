using XLSX
using HTTP
using Plots
using DataFrames
using CSV
using Statistics
using GLM

#############################################################################
                              #Data
#############################################################################

# Step 1: Define the Remote URL and Local Path
remote_url = "https://raw.githubusercontent.com/maestri-and/Bocola_Dovis-2019/refs/heads/main/data/Data.xlsx"  # Replace with your raw Git URL
local_path = "Data.xlsx"  # Local file path to save the downloaded file

# Step 2: Download the File
HTTP.open(:GET, remote_url) do file
    open(local_path, "w") do local_file
        write(local_file, read(file))
    end
end

# Step 3: Load the Excel File and transform it into a data frame
xf = XLSX.readxlsx(local_path)
println("Sheetnames: ", XLSX.sheetnames(xf))
df = DataFrame(XLSX.readtable("Data.xlsx", "Main Series"))
CSV.write("debt2output.csv", df)
rename!(df, "Debt to output ratio" => :Debt_to_output_ratio)
rename!(df, "GDP, detrended" => :GDP)
rename!(df, "Debt maturity" => :Debt_maturity)
rename!(df, "Chi shock" => :Chi)
#Rearrange the date format
# Extract the year and quarter from the Date column (e.g., "Q1-2000")
split_dates = split.(df.Date, "Q")  # Split the Date string by "-"
# Extract Year (second part of the split)
df.Year = Int64.(map(x -> parse(Int, x[1]), split_dates))
# Extract Quarter (first part of the split)
df.Quarter = map(x -> x[2], split_dates)
# Convert the Quarter to a decimal year format (e.g., Q1 = 0.0, Q2 = 0.25, Q3 = 0.5, Q4 = 0.75)
df.Time = [df.Year[i] + (parse(Int, df.Quarter[i]) - 1) * 0.25 for i in 1:nrow(df)]  # Create the Time column
df = df[(df.Time .>= 2000.0) .& (df.Time .<= 2012.25), :]

# Compute the average debt-to-GDP ratio
debt_to_gdp = df[!, :Debt_to_output_ratio]  # Defining debt to gdp
average_debt_to_gdp = mean(debt_to_gdp)  

# Compute the correlation debt-to-GDP and output
output = df[!, :GDP]  # Defining the ouput column
correlation_debt_to_gdp_output = cor(debt_to_gdp, output)

# Compute the average spread 
average_spread = mean(df[!, :Spread]) 

# Compute the standard deviation of spread
std_dev_spread = std(df[!, :Spread])

# Compute the average debt maturity
debt_maturity=df[!, :Debt_maturity]
average_maturity = mean(skipmissing(debt_maturity)) 

# Compute the standard deviation of debt maturity
std_dev_maturity = std(skipmissing(debt_maturity))

# Computing the weighting average life of Italian outstanding bonds wal_t
df_redemption = DataFrame(XLSX.readtable("Data.xlsx", "Redemption profile Italian debt")) #Read the "Redemption profile Italian debt" sheet into a DataFrame
# Extract payment columns ("t+1", "t+2", ..., "t+n")
payments = Matrix(df_redemption[:, 2:end])
# Compute wal_t for each row
N = size(payments, 2)  # Number of maturities (columns)
n = 1:N  # Maturities as 1, 2, ..., N
wal_t = [sum(n .* payments[i, :]) / sum(payments[i, :]) for i in 1:size(payments, 1)]
# Add wal_t to the DataFrame
df_redemption[!, :wal_t] = wal_t
# Merging with the main data frame
df_wal = select(df_redemption, [:Date, :wal_t])
df_merged = leftjoin(df, df_wal, on=:Date)
df_merged = df_merged[(df_merged.Time .>= 2000.0) .& (df_merged.Time .<= 2012.25), :]

# Running regression 21
## Step 1: creating interaction variables
df_merged[!, :gdp_debt] = df_merged.GDP .* df_merged.Debt_to_output_ratio  # Interaction term: gdp_t × debt_t
df_merged[!, :gdp_wal] = df_merged.GDP .* df_merged.wal_t  # Interaction term: gdp_t × wal_t
df_merged[!, :debt_chi] = df_merged.Debt_to_output_ratio .* df_merged.Chi  # Interaction term: debt_t × χ_t
df_merged[!, :gdp_chi] = df_merged.GDP .* df_merged.Chi  # Interaction term: gdp_t × χ_t
df_merged[!, :debt_wal] = df_merged.Debt_to_output_ratio .* df_merged.wal_t  # Interaction term: debt_t × wal_t
df_merged[!, :chi_wal] = df_merged.Chi .* df_merged.wal_t  # Interaction term: χ_t × wal_t
 # Drop rows with missing values
## Step 2: Converting relevant columns into floats
df_merged[!, :Spread] = float.(df_merged[!, :Spread])
df_merged[!, :GDP] = float.(df_merged[!, :GDP]) 
df_merged[!, :Debt_to_output_ratio] = float.(df_merged[!, :Debt_to_output_ratio]) 
df_merged[!, :Chi] = float.(df_merged[!, :Chi]) 
df_merged
# Step 2: Running the Regression
model = lm(@formula(Spread ~ GDP + Debt_to_output_ratio + Chi + wal_t + gdp_debt + gdp_chi + gdp_wal + debt_chi + debt_wal + chi_wal), df_merged) 

# Computing R^2 and save it in a variable
r_squared = r2(model)


#############################################################################
                              #Model
#############################################################################




#############################################################################
                              #Final Table
#############################################################################

# Create a DataFrame to represent the table
results_table = DataFrame(
    Statistic = [
        "Average debt-to-GDP ratio",
        "Correlation debt-to-GDP and output",
        "Average spread",
        "Standard deviation of spread",
        "R² of regression (21)",
        "Average debt maturity",
        "Standard deviation of debt maturity"
    ],
    Data = [
        average_debt_to_gdp,
        correlation_debt_to_gdp_output,
        average_spread,
        std_dev_spread,
        r_squared,
        average_maturity,
        std_dev_maturity
    ],
    Model = [missing, missing, missing, missing, missing, missing, missing]  # Leave Model column blank for now
)
println(results_table)