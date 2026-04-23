import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/results.csv")

# convert numeric columns
for col in ["runtime_ms", "makespan", "soc"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["runtime_s"] = df["runtime_ms"] / 1000.0

# ---------------------------
# AVERAGES per map + N
# ---------------------------
avg = df.groupby(["map", "N"]).mean(numeric_only=True).reset_index()
success_rate = df.groupby(["map", "N"])["success"].mean().reset_index()

# ---------------------------
# PLOT: Runtime
# ---------------------------
plt.figure()
for m in avg["map"].unique():
    sub = avg[avg["map"] == m]
    plt.plot(sub["N"], sub["runtime_s"], marker="o", label=m)

plt.xlabel("Number of agents (N)")
plt.ylabel("Runtime (s)")
plt.title("Runtime vs Agents (averaged over scenarios)")
plt.legend(fontsize=6)
plt.tight_layout()
plt.savefig("results/runtime.png")
plt.close()

# ---------------------------
# PLOT: Makespan
# ---------------------------
plt.figure()
for m in avg["map"].unique():
    sub = avg[avg["map"] == m]
    plt.plot(sub["N"], sub["makespan"], marker="o", label=m)

plt.xlabel("Number of agents (N)")
plt.ylabel("Makespan")
plt.title("Makespan vs Agents (averaged)")
plt.legend(fontsize=6)
plt.tight_layout()
plt.savefig("results/makespan.png")
plt.close()

# ---------------------------
# PLOT: SOC
# ---------------------------
plt.figure()
for m in avg["map"].unique():
    sub = avg[avg["map"] == m]
    plt.plot(sub["N"], sub["soc"], marker="o", label=m)

plt.xlabel("Number of agents (N)")
plt.ylabel("Sum of Costs")
plt.title("SOC vs Agents (averaged)")
plt.legend(fontsize=6)
plt.tight_layout()
plt.savefig("results/soc.png")
plt.close()

# ---------------------------
# PLOT: Success rate
# ---------------------------
plt.figure()
for m in success_rate["map"].unique():
    sub = success_rate[success_rate["map"] == m]
    plt.plot(sub["N"], sub["success"] * 100, marker="o", label=m)

plt.xlabel("Number of agents (N)")
plt.ylabel("Success rate (%)")
plt.title("Success Rate vs Agents")
plt.legend(fontsize=6)
plt.tight_layout()
plt.savefig("results/success.png")
plt.close()

print("Done. Graphs saved in results/")