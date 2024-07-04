import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot()
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid()
plt.savefig(
    "hybrid_transfomer_loss_imdb_rk4_enh.pdf",
    bbox_inches="tight",
    transparent=True,
)
