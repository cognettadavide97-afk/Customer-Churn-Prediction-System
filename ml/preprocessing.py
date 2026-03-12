from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# --- CONFIGURAZIONE PERCORSI ---
ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "Telco_customer_churn.csv"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


def clean_raw(
    df: pd.DataFrame,
    include_log_totalcharges: bool = False,
    use_gender: bool = True,
    use_total_charges: bool = True,
) -> pd.DataFrame:
    """Esegue la pulizia e il feature engineering ottimizzato (v2 configurabile)."""
    df = df.copy()

    # 1. PULIZIA NOMI E RIMOZIONE COLONNE (drop dopo il feature engineering)
    df.columns = df.columns.str.strip()
    cols_to_drop = [
        "CustomerID", "Count", "Country", "State", "City", "Lat Long",
        "Zip Code", "Churn Reason", "Churn Score", "Churn Label", "CLTV",
        "Latitude", "Longitude"
    ]

    # 2. DIGITALIZZAZIONE (MAPPING MIRATO)
    churn_map = {"Yes": 1, "No": 0, "Churned": 1, "Stayed": 0}
    if "Churn Value" in df.columns:
        df["Churn Value"] = df["Churn Value"].replace(churn_map)

    if "Gender" in df.columns:
        if use_gender:
            df["Gender"] = df["Gender"].replace({"Female": 1, "Male": 0})
        else:
            cols_to_drop.append("Gender")

    print("🎯 #Step2: Digitalizzazione variabili (Target, Gender e Servizi)")

    # 3. CONVERSIONE NUMERICA COERENTE
    num_cols = ["Total Charges", "Tenure Months", "Monthly Charges"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. FEATURE ENGINEERING (LOGICA DI BUSINESS)
    print("🛠️ #Step3: Generazione nuove Feature")

    # Calcolo spesa media evitando divisione per zero
    if {"Total Charges", "Tenure Months"}.issubset(df.columns):
        df["AvgMonthlySpend"] = df["Total Charges"] / df["Tenure Months"].replace(0, pd.NA)

    # Calcolo NumServices
    service_cols = [
        "Multiple Lines", "Online Security", "Online Backup", "Device Protection",
        "Tech Support", "Streaming TV", "Streaming Movies", "Phone Service"
    ]

    available_services = [c for c in service_cols if c in df.columns]
    service_map = {"Yes": 1, "No": 0, "No internet service": 0, "No phone service": 0}
    service_numeric = (
        df[available_services]
        .replace(service_map)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )
    df["NumServices"] = service_numeric.sum(axis=1)

    # Manteniamo solo la colonna aggregata streaming
    streaming_cols = [c for c in ["Streaming TV", "Streaming Movies"] if c in service_numeric.columns]
    if streaming_cols:
        df["StreamingBundleCount"] = service_numeric[streaming_cols].sum(axis=1)
        cols_to_drop.extend(streaming_cols)

    phone_cols = [c for c in ["Phone Service", "Multiple Lines"] if c in service_numeric.columns]
    if phone_cols:
        df["PhoneBundleCount"] = service_numeric[phone_cols].sum(axis=1)

    if "Internet Service" in df.columns:
        df["HasInternet"] = (df["Internet Service"].str.strip().str.lower() != "no").astype(int)

    if "Payment Method" in df.columns:
        df["Is_Electronic_Check"] = (
            df["Payment Method"].str.strip().str.lower() == "electronic check"
        ).astype(int)

    if include_log_totalcharges and "Total Charges" in df.columns:
        df["Log_TotalCharges"] = np.log1p(pd.to_numeric(df["Total Charges"], errors="coerce"))

    # Toggle per tenere/rimuovere Total Charges
    if not use_total_charges and "Total Charges" in df.columns:
        cols_to_drop.append("Total Charges")

    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    print("#Step1: Pulizia intestazioni e rimozione colonne irrilevanti")
    return df.drop_duplicates()


def split_save(df: pd.DataFrame):
    """Gestisce lo split dei dati e il salvataggio fisico."""
    X = df.drop(columns=["Churn Value"])
    y = df["Churn Value"]

    # Split stratificato per mantenere le proporzioni del Churn
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Salvataggio CSV
    pd.concat([X_train, y_train], axis=1).to_csv(PROC_DIR / "train_raw.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(PROC_DIR / "test_raw.csv", index=False)
    print(f"💾 #Step4: File salvati con successo in {PROC_DIR}")

    # --- VERIFICA SUL FILE SALVATO (TRAIN_RAW) ---
    print(f"\n🧪 #CHECK: Validazione dati su {'train_raw.csv'}")
    check_df = pd.read_csv(PROC_DIR / "train_raw.csv")

    print("\n📊 1. INFO - Verifica tipi e non-nulli:")
    check_df.info()

    print("\n📈 2. DESCRIBE - Statistiche ogni singola colonna:")
    print(check_df.describe().T)

    print("\n👀 3. HEAD - Anteprima record digitalizzati:")
    print(check_df.head())

    print("PROC_DIR:", PROC_DIR)
    print("Columns in file:", [repr(c) for c in check_df.columns if "Streaming" in c])

    print("\n" + "—" * 50)


def main():
    print(f"🚀 #START: Processing {RAW_FILE.name}")
    try:
        if not RAW_FILE.exists():
            print("❌ Errore: File non trovato!")
            return

        # Config v2: cambia qui i toggle
        use_gender = False
        use_total_charges = True

        df = pd.read_csv(RAW_FILE)
        df_processed = clean_raw(
            df,
            include_log_totalcharges=False,
            use_gender=use_gender,
            use_total_charges=use_total_charges,
        )
        split_save(df_processed)

        print("\n" + "=" * 30)
        print(f"✅ DATASET PRONTO: {df_processed.shape[0]} righe, {df_processed.shape[1]} colonne")
        print(f"⚙️ Config -> use_gender={use_gender}, use_total_charges={use_total_charges}")
        print("=" * 30)

    except Exception as e:
        print(f"❌ #ERROR: {e}")


if __name__ == "__main__":
    main()