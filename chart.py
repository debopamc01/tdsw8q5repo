# chart.py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_synthetic_data(n_customers: int = 300, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    # Acquisition channels with realistic mix
    channels = ["Paid Search", "Social", "Email", "Affiliate", "Display"]
    channel_probs = [0.30, 0.25, 0.20, 0.15, 0.10]

    channel = rng.choice(channels, size=n_customers, p=channel_probs)

    # Base CAC (Customer Acquisition Cost) by channel (in $)
    cac_means = {
        "Paid Search": 250,
        "Social": 180,
        "Email": 80,
        "Affiliate": 150,
        "Display": 120,
    }
    cac_std = {
        "Paid Search": 40,
        "Social": 35,
        "Email": 20,
        "Affiliate": 30,
        "Display": 25,
    }

    acquisition_cost = []
    for ch in channel:
        cac = rng.normal(cac_means[ch], cac_std[ch])
        # Keep CAC in a realistic positive band
        cac = np.clip(cac, 40, 400)
        acquisition_cost.append(cac)
    acquisition_cost = np.array(acquisition_cost)

    # Lifetime value model (in $)
    # LTV increases with CAC but with diminishing returns + noise
    # Base model: LTV = a * CAC - b * CAC^2 + segment + noise
    a = 5.5
    b = 0.004   # diminishing returns

    # Channel uplift factor (better-quality channels drive higher LTV)
    channel_ltv_uplift = {
        "Paid Search": 1.10,
        "Social": 1.05,
        "Email": 1.00,
        "Affiliate": 1.03,
        "Display": 0.95,
    }

    base_ltv = a * acquisition_cost - b * (acquisition_cost ** 2)
    noise = rng.normal(0, 80, size=n_customers)

    lifetime_value = []
    for cac, ch, base in zip(acquisition_cost, channel, base_ltv):
        uplift = channel_ltv_uplift[ch]
        ltv = base * uplift + noise[rng.integers(0, len(noise))]
        # Ensure LTV is at least somewhat reasonable
        ltv = np.clip(ltv, 50, 3000)
        lifetime_value.append(ltv)
    lifetime_value = np.array(lifetime_value)

    # Compute LTV/CAC ratio as ROI metric
    ltv_to_cac = lifetime_value / acquisition_cost

    # ROI tiers for executive-level storytelling
    roi_tier = pd.cut(
        ltv_to_cac,
        bins=[0, 1.5, 3, np.inf],
        labels=["Underperforming (LTV < 1.5× CAC)",
                "Viable (1.5×–3× CAC)",
                "High Return (>3× CAC)"],
    )

    df = pd.DataFrame({
        "customer_id": np.arange(1, n_customers + 1),
        "acquisition_cost": acquisition_cost,
        "lifetime_value": lifetime_value,
        "channel": channel,
        "ltv_to_cac": ltv_to_cac,
        "roi_tier": roi_tier,
    })

    return df


def create_chart(df: pd.DataFrame, output_path: str = "chart.png") -> None:
    # Seaborn styling for executive / presentation context
    sns.set_style("whitegrid")
    sns.set_context("talk")  # larger fonts for presentations

    # 8x8 inches at 64 dpi -> 512x512 pixels
    plt.figure(figsize=(8, 8), dpi=64)

    # Scatterplot: LTV vs CAC by channel and ROI tier
    ax = sns.scatterplot(
        data=df,
        x="acquisition_cost",
        y="lifetime_value",
        hue="channel",
        style="roi_tier",
        s=80,
        alpha=0.9,
        edgecolor="black",
    )

    # Reference line at LTV = 3 × CAC (common profitability benchmark)
    x_vals = np.linspace(df["acquisition_cost"].min(), df["acquisition_cost"].max(), 100)
    y_benchmark = 3 * x_vals
    plt.plot(
        x_vals,
        y_benchmark,
        linestyle="--",
        linewidth=1.5,
        label="LTV = 3× CAC benchmark",
    )

    # Titles and labels
    ax.set_title(
        "Customer Lifetime Value vs Acquisition Cost\n"
        "by Marketing Channel and ROI Tier",
        pad=20,
    )
    ax.set_xlabel("Customer Acquisition Cost (USD)")
    ax.set_ylabel("Customer Lifetime Value (USD)")

    # Legend handling: combine scatter and line
    handles, labels = ax.get_legend_handles_labels()
    # Remove default 'roi_tier' title if present in labels, keep all items.
    ax.legend(
        handles=handles,
        labels=labels,
        title="Channel & ROI Tier",
        frameon=True,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.0,
    )

    # Clean up axes for a professional look
    sns.despine(offset=10, trim=True)

    plt.tight_layout()

    # Save exactly 512x512 pixels
    plt.savefig(output_path, dpi=64)
    plt.close()


if __name__ == "__main__":
    data = generate_synthetic_data()
    create_chart(data, "chart.png")
