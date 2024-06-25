import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(
    label="Classical RK4_ENH Training Loss",
    color="blue",
    linestyle="--",
)
ax.plot(
    
    label="Classical RK4_ENH Testing Loss",
    color="blue",
)

ax.plot(
    label="Quantum RK4_ENH Training Loss",
    color="red",
    linestyle="--",
)
ax.plot(
    label="Quantum RK4_ENH Testing Loss",
    color="red",
)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid()
plt.savefig(
    "hybrid_transfomer_loss_imdb_rk4_enh.pdf",
    bbox_inches="tight",
    transparent=True,
)
