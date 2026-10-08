"""
Train & evaluate the Linear Regression model from the command line.

    python train.py                      # uses data/housing_data.csv
    python train.py --data "<folder>"    # looks for 'Project housing data' (.csv / .xlsx) in a folder
"""
import argparse
import json
from pathlib import Path

from src.house_model import (SAMPLE_HOUSE, find_data_file, predict_with_explanation,
                             train_model)


def main() -> None:
    parser = argparse.ArgumentParser(description="House Price Prediction - Linear Regression")
    parser.add_argument("--data", help="Folder containing 'Project housing data' (.csv/.xlsx)")
    args = parser.parse_args()

    data_file = find_data_file(args.data)
    print(f"Dataset: {data_file}")
    bundle = train_model(data_file)
    m = bundle.metrics

    print(f"\nRows: {m['n_rows_raw']} raw -> {m['n_rows']} clean | missing values: {m['missing_values_raw']}")
    print(f"Train/Test split: {m['n_train']} / {m['n_test']}")
    print(f"Features after One-Hot encoding: {bundle.feature_names}")

    print("\n=== Test-set metrics ===")
    t = m["test"]
    print(f"R²   : {t['r2']:.4f}  (train {m['train']['r2']:.4f})")
    print(f"RMSE : ${t['rmse']:,.0f}")
    print(f"MAE  : ${t['mae']:,.0f}")
    print(f"MAPE : {m['mape']:.1%}")

    print("\n=== Coefficients (sorted by standardized importance) ===")
    print(bundle.coefficients[["feature", "coef_std", "coef_real"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

    print("\n=== Sample prediction ===")
    print(SAMPLE_HOUSE)
    r = predict_with_explanation(bundle, SAMPLE_HOUSE)
    print(f"Predicted price : ${r['prediction']:,.0f}  (range ±RMSE: ${r['low']:,.0f} – ${r['high']:,.0f})")
    print(f"Average house   : ${r['baseline']:,.0f}")
    print("Contributions vs. average house:")
    for feat, val in r["contributions"].items():
        print(f"  {feat:<14} {val:+,.0f}")
    print(f"Price per sqft  : ${r['price_per_sqft']:,.0f} (market median ${r['market_price_per_sqft']:,.0f})")
    print(f"Similar homes   : {r['similar_count']} with median ${r['similar_median']:,.0f} "
          f"(IQR ${r['similar_p25']:,.0f} – ${r['similar_p75']:,.0f})")

    out = Path("reports")
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    bundle.coefficients.to_csv(out / "coefficients.csv", index=False)
    print(f"\nSaved reports to {out.resolve()}")


if __name__ == "__main__":
    main()
