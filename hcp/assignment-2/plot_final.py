import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys

if len(sys.argv) < 2:
    print("Usage: python plot_final.py <data_file>")
    sys.exit(1)

# Load Data
df = pd.read_csv(sys.argv[1])

# Setup styling
sns.set_theme(style="whitegrid")
plt.figure(figsize=(18, 10))

# Create a FacetGrid to separate by List Size
# We want to see how Time changes with Thread Count for each Method, per Size.
g = sns.FacetGrid(df, col="Size", col_wrap=2, height=4, aspect=1.5, sharey=False)

# Map the plotting function
# X-axis: Threads, Y-axis: Time, Hue: Method
g.map_dataframe(sns.lineplot, x="Threads", y="Time", hue="Method", marker="o", linewidth=2.5)

# Add titles and legends
g.add_legend(title="Method")
g.set_axis_labels("Number of Threads", "Execution Time (s)")
g.set_titles("List Size: {col_name}")

# Adjust layout
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Parallel Sorting Check: Performance Scaling by Thread Count', fontsize=16)

# Save
plt.savefig('final_scaling_analysis.png')
print("Plot saved as final_scaling_analysis.png")