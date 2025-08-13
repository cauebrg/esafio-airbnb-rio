import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    base_dir = Path(".")
    listings_path = base_dir / "listings_cleaned.csv"
    reviews_path = base_dir / "reviews.csv"
    out_dir = base_dir / "airbnb_rio_outputs"
    out_dir.mkdir(exist_ok=True)

    # 1) Ler arquivos
    df_listings = pd.read_csv(listings_path)
    df_reviews = pd.read_csv(reviews_path)

    # 2) Merge por 'id'
    df = pd.merge(df_listings, df_reviews, on="id", how="inner")
    df_raw = df.copy()

    # 3) Ajuste de tipos e tratamento de nulos
    numeric_cols = ["accommodates", "bathrooms", "bedrooms", "beds", "price",
                    "number_of_reviews", "review_scores_rating"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["neighbourhood_cleansed", "room_type"]:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("Unknown")

    for col in numeric_cols:
        med = df[col].median(skipna=True)
        df[col] = df[col].fillna(med)

    # 4) Remover outliers de price usando IQR
    def iqr_bounds(s):
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return lower, upper, q1, q3, iqr

    price_lower, price_upper, q1, q3, iqr = iqr_bounds(df["price"])

    plt.figure()
    plt.boxplot(df_raw["price"].dropna())
    plt.title("Boxplot de Preços (Antes do IQR)")
    plt.savefig(out_dir / "boxplot_price_before.png", bbox_inches="tight")
    plt.close()

    df = df[(df["price"] >= price_lower) & (df["price"] <= price_upper)].copy()

    plt.figure()
    plt.boxplot(df["price"].dropna())
    plt.title("Boxplot de Preços (Depois do IQR)")
    plt.savefig(out_dir / "boxplot_price_after.png", bbox_inches="tight")
    plt.close()

    # 5) Codificar variáveis categóricas
    for col in ["room_type", "neighbourhood_cleansed"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
            df[f"{col}_code"] = df[col].cat.codes

    # 6) Normalização (z-score)
    def zscore(s):
        mu = s.mean()
        sd = s.std(ddof=0)
        return (s - mu) / sd if sd and not np.isnan(sd) else 0

    for col in numeric_cols:
        if col in df.columns:
            df[f"{col}_z"] = zscore(df[col])

    # 7) Salvar CSV final
    df.to_csv(out_dir / "airbnb_rio_cleaned.csv", index=False)
    print("✅ Arquivo limpo salvo em:", out_dir / "airbnb_rio_cleaned.csv")

if __name__ == "__main__":
    main()

# === PACOTE FINAL ===
from pathlib import Path
out_dir = Path("airbnb_rio_outputs")
df_full = pd.read_csv(out_dir / "airbnb_rio_cleaned.csv")

# (A) Dataset de modelagem: só o essencial (ajuste se quiser)
cols_keep = [
    "id",
    "neighbourhood_cleansed", "room_type",
    "accommodates","bathrooms","bedrooms","beds","price",
    "number_of_reviews","review_scores_rating",
    "neighbourhood_cleansed_code","room_type_code"
]
# Se alguma coluna não existir, ignorar silenciosamente
cols_keep = [c for c in cols_keep if c in df_full.columns]
df_model = df_full[cols_keep].copy()

# Remover colunas redundantes (ex.: versões auxiliares que você não quer no treino)
# Aqui, mantemos os códigos categóricos e os originais de texto (para leitura/relatório).
# Se quiser 100% numérico, drope as strings:
# df_model = df_model.drop(columns=["neighbourhood_cleansed","room_type"], errors="ignore")

# Salvar artefatos finais
df_model.to_csv(out_dir / "airbnb_rio_model.csv", index=False)

# Resumo de sanidade (para o avaliador)
summary = []
summary.append(f"REGISTROS (full): {len(df_full):,}")
summary.append(f"REGISTROS (model): {len(df_model):,}")
summary.append("\nCOLUNAS (model): " + ", ".join(df_model.columns))

with open(out_dir / "README_ENTREGA.txt", "w", encoding="utf-8") as f:
    f.write(
        "Desafio Airbnb RJ – Entrega Final\n"
        "\nArquivos:\n"
        "- airbnb_rio_cleaned.csv  (dataset tratado + normalizações)\n"
        "- airbnb_rio_model.csv    (dataset enxuto p/ modelagem)\n"
        "- boxplot_price_before.png / boxplot_price_after.png\n"
        "- eda_summary.txt (opcional)\n\n"
        "Notas:\n"
        "- Outliers de price tratados via IQR.\n"
        "- Categóricas convertidas com .astype('category').cat.codes.\n"
        "- Numéricos normalizados (z-score) disponíveis no CSV enriquecido.\n"
    )

print("✅ Entrega final gerada: airbnb_rio_outputs/airbnb_rio_model.csv e README_ENTREGA.txt")
