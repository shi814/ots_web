import pandas as pd
import numpy as np
import os


def sellmeier_n(wavelength_um, K1, K2, K3, L1, L2, L3):
    lam2 = wavelength_um ** 2
    n2 = (
            1
            + K1 * lam2 / (lam2 - L1)
            + K2 * lam2 / (lam2 - L2)
            + K3 * lam2 / (lam2 - L3)
    )
    return np.sqrt(n2)


def main():
    input_path = os.path.join("glass", "schott_glass_pars.xlsx")

    # 没有表头，所以用 header=None
    df = pd.read_excel(input_path, header=None)

    # 手动指定列名：第1列是 Glass，后面依次 K1,K2,K3,L1,L2,L3
    df.columns = ["Glass", "K1", "K2", "K3", "L1", "L2", "L3"]

    wavelengths_nm = [450, 580, 750]
    wavelengths_um = [w / 1000.0 for w in wavelengths_nm]

    for w_nm, w_um in zip(wavelengths_nm, wavelengths_um):
        col_name = f"n_{w_nm}nm"
        df[col_name] = df.apply(
            lambda row: sellmeier_n(
                w_um,
                row["K1"], row["K2"], row["K3"],
                row["L1"], row["L2"], row["L3"],
            ),
            axis=1
        )

    output_path = os.path.join("glass", "schott_glass_with_n.xlsx")
    df.to_excel(output_path, index=False)
    print(f"Done. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
