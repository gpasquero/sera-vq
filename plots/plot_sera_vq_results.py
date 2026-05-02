import matplotlib.pyplot as plt


def main():
    # =========================
    # DATA (tus resultados)
    # =========================
    pca = [
        (32, 0.8077),
        (64, 0.8444),
        (96, 0.8560),
        (128, 0.8608),
        (192, 0.8667),
        (256, 0.8685),
        (384, 0.8687),
        (1536, 0.8672),
    ]

    vq = [
        (8, 0.7467),
        (12, 0.7926),
        (16, 0.8150),
        (24, 0.8333),
        (32, 0.8430),
    ]

    pca_x, pca_y = zip(*pca)
    vq_x, vq_y = zip(*vq)

    # =========================
    # STYLE (paper clean)
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
    # HIGHLIGHT REGION (clave)
    # =========================
    plt.axvspan(8, 40, alpha=0.07)
    plt.text(10, 0.858, "Discrete regime", fontsize=10)

    # =========================
    # ANNOTATIONS (key insight)
    # =========================
    plt.scatter([32], [0.8430], s=90, zorder=5)
    plt.annotate(
        "SERA-VQ\n32B, 0.843",
        xy=(32, 0.8430),
        xytext=(55, 0.83),
        arrowprops=dict(arrowstyle="->"),
    )

    plt.scatter([32], [0.8077], s=90, zorder=5)
    plt.annotate(
        "PCA\n32B, 0.808",
        xy=(32, 0.8077),
        xytext=(60, 0.795),
        arrowprops=dict(arrowstyle="->"),
    )

    # =========================
    # AXES
    # =========================
    plt.xscale("log", base=2)
    plt.xlabel("Memory per embedding (bytes, log scale)")
    plt.ylabel("Spearman correlation (STS-B)")
    plt.ylim(0.78, 0.87)

    # =========================
    # TITLE (paper-safe)
    # =========================
    plt.title(
        "Discrete Codes Outperform Dense Embeddings in Low-Memory Regimes"
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
    plt.savefig("sera_vq_paper.png", dpi=300)
    plt.savefig("sera_vq_paper.pdf")

    print("Saved:")
    print("  sera_vq_paper.png")
    print("  sera_vq_paper.pdf")


if __name__ == "__main__":
    main()
