import pandas as pd
import scipy.stats as stats
import scipy.special as sps


def check_normality(data: pd.Series):
    freq_table = data.value_counts().sort_index().reset_index(name="Fabs")
    freq_table.columns = ["Value", "Fabs"]
    freq_table["Fac"] = freq_table["Fabs"].cumsum()
    freq_table["Frac"] = freq_table["Fac"] / freq_table["Fac"].iloc[-1]

    z_scores = (freq_table["Value"] - data.mean()) / data.std()
    freq_table["Zi"] = z_scores
    freq_table["FracEsp"] = sps.ndtr(z_scores)

    frac = freq_table["Frac"]
    frac_esp = freq_table["FracEsp"]
    D_neg = (frac_esp - frac).abs()
    D_pos = frac_esp - frac.shift(fill_value=0)
    D = max(D_neg.max(), D_pos.max())

    n = len(data)
    alpha = 0.05
    D_critical = stats.ksone.ppf(1 - alpha / 2, n)
    result = "seguem" if D < D_critical else "não seguem"
    print(f"Os dados {result} uma distribuição normal.")


def check_distribution(distributions, data: pd.Series):
    results = []

    for distribution in distributions:
        dist = getattr(stats, distribution)
        params = dist.fit(data)
        args = params if distribution != "norm" else ()
        D, p = stats.kstest(data, distribution, args=args,
                            alternative="greater" if distribution == "norm" else "two-sided")
        results.append((distribution, D, p, D < p))

    df = pd.DataFrame(results, columns=["Distribution", "Distance", "p_value", "D < p"])
    df.sort_values("p_value", ascending=False, inplace=True)

    print("\nDistributions sorted by goodness of fit:")
    print("----------------------------------------")
    print(df)
