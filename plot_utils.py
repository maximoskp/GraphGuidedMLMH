import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_idioms_1(data_tsne, ids_np):
    # Ensure 1D labels
    labels = ids_np.squeeze()

    # Unique styles
    unique_styles = np.unique(labels)
    num_styles = len(unique_styles)

    plt.figure(figsize=(8, 6))

    # Use seaborn color palette
    # palette = sns.color_palette("tab10", num_styles)
    palette = sns.color_palette("hls", num_styles)

    for i, style in enumerate(unique_styles):
        mask = labels == style
        plt.scatter(
            data_tsne[mask, 0],
            data_tsne[mask, 1],
            s=40,
            alpha=0.8,
            color=palette[i],
            label=f"Style {style}"
        )

    plt.title("t-SNE of LSTM Idiom Embeddings")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.tight_layout()
    plt.show()
# end plot_idioms_1

def plot_idioms_2(data_tsne, ids_np, caption=''):
    labels = ids_np.squeeze()

    df = pd.DataFrame({
        "tsne_1": data_tsne[:, 0],
        "tsne_2": data_tsne[:, 1],
        "style": labels
    })

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="tsne_1",
        y="tsne_2",
        hue="style",
        palette="hls",
        s=30,
        alpha=0.7
    )

    plt.title(f"t-SNE of {caption} Idiom Embeddings")
    plt.legend(title="Style")
    plt.tight_layout()
    os.makedirs("figs", exist_ok=True)
    plt.savefig(f"figs/tsne_{caption}.png", dpi=300)
    plt.show()
# end plot_idioms_2

