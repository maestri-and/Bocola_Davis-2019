using XLSX
using HTTP
using Plots
using DataFrames
using CSV
using Statistics
using StatsBase

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

spread_data = df_merged[!, :Spread]


#############################################################################
                              #Model
#############################################################################


#############################################################################
                              #Graph
#############################################################################

# Define bins (intervals) and labels
bins = [0, 0.75, 1.5, 2.25, Inf]  # Last bin is open-ended (>2.25)
bin_labels = ["[0, 0.75)", "[0.75, 1.5)", "[1.5, 2.25)", ">2.25"]

# Create a function to assign bin labels manually, including small negative spreads in the first bin
function assign_bins(data, bins, labels)
    assigned_bins = String[]  # Initialize an empty vector for bin labels
    for value in data
        if isnan(value)
            push!(assigned_bins, "NaN")  # Handle NaN values if necessary
        elseif -0.5 <= value < 0
            push!(assigned_bins, labels[1])  # Assign small negative values to the first bin
        else
            bin_assigned = false  # Track if the value has been assigned to a bin
            for i in 1:(length(bins) - 1)
                if bins[i] <= value < bins[i + 1]
                    push!(assigned_bins, labels[i])
                    bin_assigned = true
                    break
                end
            end
            # Handle the last open-ended bin
            if !bin_assigned && value >= bins[end - 1]
                push!(assigned_bins, labels[end])
            end
        end
    end
    return assigned_bins
end

# Assign bins to spread data
binned_data = assign_bins(spread_data, bins, bin_labels)

# Count the frequency of each bin
frequencies = countmap(binned_data)

# Normalize frequencies to represent proportions
total = sum(values(frequencies))
normalized_frequencies = [frequencies[label] / total for label in bin_labels]

# Placeholder values for "Model" data (approximated from screenshot)
model_frequencies = [0.7, 0.2, 0.05, 0.05]  # Approximated values to be replaced

# Define x-axis positions for side-by-side bars
x_data = collect(1:length(bin_labels))  # Positions for Data bars
x_model = x_data .+ 0.3  # Shift Model bars slightly to the right

# Plot the histogram with both "Data" and "Model" bars
bar(x_data, normalized_frequencies, label="Data", bar_width=0.3, color=:blue, xlabel="Interest rate spreads", ylabel="Frequency", xticks=(x_data, bin_labels), yticks=0:0.1:0.8, ylims=(0, 0.8), title="Panel A. Interest rate spreads distribution")
bar!(x_model, model_frequencies, label="Model", bar_width=0.3, color=:gray)

# Save the plot
savefig("interest_rate_spreads.png")