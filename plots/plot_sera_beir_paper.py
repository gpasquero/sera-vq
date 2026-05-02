import matplotlib.pyplot as plt


def main():
    # =========================
    # DATA: BEIR SciFact results
    # Metric: nDCG@10
    # =========================

    pca = [
        (32, 0.4510),
        (64, 0.5408),
        (96, 0.5957),
        (128, 0.6118),
    ]

    vq = [
        (8, 0.3657),
        (16, 0.4869),
        (24, 0.5287),
        (32, 0.5599),
    ]

    pca_x, pca_y = zip(*pca)
    vq_x, vq_y = zip(*vq)

    # =========================
    # STYLE
    # =========================
    plt.figure(figsize=(8, 5))
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
    })

    # =========================
    # PLOTS
    # =========================
    plt.plot(
        pca_x,
        pca_y,
        marker="o",
        linewidth=2,
        label="PCA + int8 (dense)",
    )

    plt.plot(
        vq_x,
        vq_y,
        marker="o",
        linewidth=2,
        label="SERA-VQ (discrete codes)",
    )

    # =========================
    # HIGHLIGHT LOW-MEMORY REGION
    # =========================
    plt.axvspan(8, 40, alpha=0.07)
    plt.text(10, 0.605, "Low-memory regime", fontsize=10)

    # =========================
    # KEY ANNOTATIONS
    # =========================
    plt.scatter([32], [0.5599], s=90, zorder=5)
    plt.annotate(
        "SERA-VQ\n32B, 0.560",
        xy=(32, 0.5599),
        xytext=(43, 0.535),
        arrowprops=dict(arrowstyle="->"),
    )

    plt.scatter([32], [0.4510], s=90, zorder=5)
    plt.annotate(
        "PCA\n32B, 0.451",
        xy=(32, 0.4510),
        xytext=(43, 0.43),
        arrowprops=dict(arrowstyle="->"),
    )

    # Optional: mark PCA-64 comparison
    plt.scatter([64], [0.5408], s=70, zorder=5)
    plt.annotate(
        "PCA\n64B, 0.541",
        xy=(64, 0.5408),
        xytext=(75, 0.505),
        arrowprops=dict(arrowstyle="->"),
        fontsize=10,
    )

    # =========================
    # AXES
    # =========================
    plt.xscale("log", base=2)
    plt.xlabel("Memory per embedding (bytes, log scale)")
    plt.ylabel("nDCG@10 on BEIR SciFact")
    plt.ylim(0.34, 0.63)

    # =========================
    # TITLE
    # =========================
    plt.title(
        "Discrete Codes Improve Retrieval Quality Under Extreme Compression"
    )

    # =========================
    # GRID / LEGEND
    # =========================
    plt.grid(True, alpha=0.15)
    plt.legend(loc="lower right")

    plt.tight_layout()

    # =========================
    # SAVE
    # =========================
    plt.savefig("sera_beir_scifact_paper.png", dpi=300)
    plt.savefig("sera_beir_scifact_paper.pdf")

    print("Saved:")
    print("  sera_beir_scifact_paper.png")
    print("  sera_beir_scifact_paper.pdf")


if __name__ == "__main__":
    main()
