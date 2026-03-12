import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# --- CONFIGURAZIONE GLOBALE ---
plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.edgecolor': '#333333',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'legend.frameon': False
    
})
PALETTE = {"0": "#2a9d8f", "1": "#e76f51"}


CHART_REVIEW = {
    "importanti": {
        "churn_distribution_percent.png": "KPI base: quantifica il fenomeno churn e lo sbilanciamento classi.",
        "Churn_Rate_by_Contract_Type.png": "Insight azionabile: il tipo di contratto guida la strategia di retention.",
        "churn_rate_grouped_by_tenure_group.png": "Mostra quando avviene il rischio (fase iniziale della relazione cliente).",
        "Churn_Heatmap_Grid.png": "Vista sintetica rischio costo/anzianità, utile per prioritizzare campagne.",
        "InternetServicevsChurn.png": "Segmentazione commerciale: individua offerte/servizi ad alto rischio.",
        "PaymentMethodvsChurn.png": "Segmentazione pagamenti: evidenzia cluster con maggiore churn.",
        "Scatter_Tenure_vs_Charges.png": "Utile in esposizione per raccontare la 'red zone' clienti ad alto rischio.",
        "map_churn_distribution.png": "Geografia del churn: supporta eventuali azioni territoriali (da dati raw).",
    },
    "secondari_o_da_scartare_in_presentazione_executive": {
        "full_correlation_matrix_pearson.png": "Molto tecnico e denso: utile in appendice, non in executive summary.",
        "boxplots_outliers.png": "Grafico di data-quality, meno rilevante per decisioni business immediate.",
        "charges_per_service_kde.png": "Derivata utile solo se si discute pricing avanzato.",
        "demographic_churn.png": "Può essere sensibile/meno azionabile senza una strategia segment-specific.",
        "economic_value_distribution.png": "Buono ma ridondante se già si mostrano costi+tenure+contract.",
    },
}


def export_chart_comments(output_path: Path) -> None:
    """Esporta una guida sintetica dei grafici utili per esposizione aziendale."""
    comment_file = output_path / "commenti_grafici.txt"

    lines = [
        "GUIDA RAPIDA GRAFICI CHURN",
        "=" * 40,
        "",
        "GRAFICI PRIORITARI (consigliati in esposizione):",
    ]
    for chart, note in CHART_REVIEW["importanti"].items():
        lines.append(f"- {chart}: {note}")

    lines.extend([
        "",
        "GRAFICI SECONDARI / DA SCARTARE in executive summary:",
    ])
    for chart, note in CHART_REVIEW["secondari_o_da_scartare_in_presentazione_executive"].items():
        lines.append(f"- {chart}: {note}")

    lines.extend([
        "",
        "Nota: i grafici secondari restano utili in appendice tecnica o sessioni di approfondimento.",
    ])

    comment_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Commenti grafici salvati in: {comment_file}")

def harmonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Allinea i nomi colonna tra raw e processed con cambi minimi."""
    df = df.copy()
    rename_map = {
        "Churn Value": "ChurnValue",
        "Tenure Months": "TenureMonths",
        "Monthly Charges": "MonthlyCharges",
        "Total Charges": "TotalCharges",
        "AvgMonthlySpend": "Avg Monthly Spend",
        "Senior Citizen": "SeniorCitizen",
        "Phone Service": "PhoneService",
        "Multiple Lines": "MultipleLines",
        "Internet Service": "InternetService",
        "Online Security": "OnlineSecurity",
        "Online Backup": "OnlineBackup",
        "Device Protection": "DeviceProtection",
        "Tech Support": "TechSupport",
        "Streaming TV": "StreamingTV",
        "Streaming Movies": "StreamingMovies",
        "Paperless Billing": "PaperlessBilling",
        "Payment Method": "PaymentMethod",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if "ChurnValue" in df.columns:
        df["ChurnValue"] = df["ChurnValue"].astype(str)
        df["Churn Value"] = df["ChurnValue"]
    if "Charges per Service" not in df.columns and {"MonthlyCharges", "NumServices"}.issubset(df.columns):
        denom = pd.to_numeric(df["NumServices"], errors="coerce").replace(0, np.nan)
        monthly = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
        df["Charges per Service"] = (monthly / denom).fillna(monthly)
    return df
# --- FUNZIONI DI VISUALIZZAZIONE ---
def plot_churn_distribution(df, output_path):
    """
    Mostra la distribuzione percentuale del Churn.
    Insight: Permette di capire se il dataset è sbilanciato e qual è il tasso di abbandono base.
    """
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x="ChurnValue", hue="ChurnValue", data=df, palette=PALETTE, legend=False)
    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        ax.annotate(percentage, (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=12, weight='bold', xytext=(0, 5),
                    textcoords='offset points')
    plt.title("Distribuzione Generale del Churn (%)")
    plt.savefig(output_path / "churn_distribution_percent.png", dpi=300, bbox_inches='tight')
    plt.close()
def plot_tenure_kde(df, output_path):
    """
    Analisi densità Tenure vs Churn.
    Insight: Chi abbandona tende ad avere una tenure molto bassa (picco nei primi mesi).
    """
    plt.figure(figsize=(10, 5))
    for val, label in [("0", "Stayed"), ("1", "Churned")]:
        sns.kdeplot(data=df[df["Churn Value"] == val], x="TenureMonths", 
                    fill=True, label=label, color=PALETTE[val], alpha=0.5)
    plt.title("Distribuzione della Tenure: Clienti Fedeli vs Churned")
    plt.legend()
    plt.savefig(output_path / "Tenure_KDE_Distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
def plot_monthly_charges_kde(df, output_path):
    """
    Analisi densità Costi Mensili vs Churn.
    Insight: I clienti con costi mensili più elevati mostrano una densità di churn maggiore.
    """
    plt.figure(figsize=(10, 5))
    for val, label in [("0", "Stayed"), ("1", "Churned")]:
        sns.kdeplot(data=df[df["Churn Value"] == val], x="MonthlyCharges", 
                    fill=True, label=label, color=PALETTE[val], alpha=0.5)
    plt.title("Impatto dei Costi Mensili sul Churn")
    plt.legend()
    plt.savefig(output_path / "MonthlyCharges_KDE_Distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
def plot_contract_churn(df, output_path):
    """
    Confronto contratti e Churn Rate.
    Insight: Il contratto 'Month-to-month' è il principale predittore di churn.
    """
    # Countplot
    plt.figure(figsize=(10, 6))
    sns.countplot(x="Contract", hue="ChurnValue", data=df, palette=PALETTE)
    plt.title("Contract vs Churn")
    plt.savefig(output_path / "ContractvsChurn.png", dpi=300)
    plt.close()
    # Barplot Churn Rate
    temp_df = df.copy()
    temp_df["ChurnValue"] = temp_df["ChurnValue"].astype(int)
    churn_rate = temp_df.groupby("Contract")["ChurnValue"].mean().reset_index()
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x="Contract", y="ChurnValue", data=churn_rate, color="#264653")
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1%}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points', weight='bold')
    plt.title("Churn Rate by Contract Type")
    plt.savefig(output_path / "Churn_Rate_by_Contract_Type.png", dpi=300)
    plt.close()
def plot_services_and_payment(df, output_path):
    """
    Analisi Internet Service e Metodo di Pagamento.
    Insight: Fibra Ottica ed Electronic Check sono associati a churn più elevati.
    """
    # Internet Service
    plt.figure(figsize=(10, 6))
    sns.countplot(x="InternetService", hue="ChurnValue", data=df, palette=PALETTE)
    plt.title("Internet Service vs Churn")
    plt.savefig(output_path / "InternetServicevsChurn.png", dpi=300)
    plt.close()
    # Payment Method
    plt.figure(figsize=(12, 6))
    sns.countplot(x="PaymentMethod", hue="ChurnValue", data=df, palette=PALETTE)
    plt.xticks(rotation=15)
    plt.title("Payment Method vs Churn")
    plt.savefig(output_path / "PaymentMethodvsChurn.png", dpi=300, bbox_inches='tight')
    plt.close()
def plot_map_distribution(df, output_path):
    """
    Mappa geografica del Churn.
    Insight: Identifica se ci sono cluster geografici di abbandono (hotspots).
    """
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x="Longitude", y="Latitude", hue="ChurnValue", data=df,
                    alpha=0.4, palette=PALETTE, s=15)
    plt.title("Customer Location and Churn")
    plt.savefig(output_path / "map_churn_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
def plot_tenure_group_rate(df, output_path):
    """
    Churn Rate raggruppato per fasce di anzianità.
    Insight: Il churn cala drasticamente dopo il primo anno (0-12 mesi).
    """
    df["Tenure Group"] = pd.cut(df["TenureMonths"].astype(float), bins=[0, 12, 24, 48, 72],
                                 labels=["0-12", "12-24", "24-48", "48-72"])
    temp_df = df.copy()
    temp_df["ChurnValue"] = temp_df["ChurnValue"].astype(int)
    tenure_churn = temp_df.groupby("Tenure Group", observed=True)["ChurnValue"].mean().reset_index()
    plt.figure(figsize=(8, 5))
    sns.barplot(x="Tenure Group", y="ChurnValue", data=tenure_churn, color="#457b9d")
    plt.title("Churn Rate by Tenure Group")
    plt.savefig(output_path / "churn_rate_grouped_by_tenure_group.png", dpi=300)
    plt.close()
def plot_scatter_tenure_charges(df, output_path):
    """
    Scatter plot Tenure vs Monthly Charges.
    Insight: Identifica la 'Red Zone' (clienti nuovi ad alto costo).
    """
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x="TenureMonths", y="MonthlyCharges", hue="Churn Value", 
                    palette=PALETTE, alpha=0.4, s=30, edgecolor=None)
    plt.axvline(x=12, color='grey', linestyle='--', alpha=0.5)
    plt.axhline(y=df["MonthlyCharges"].mean(), color='grey', linestyle='--', alpha=0.5)
    plt.title("Analisi Congiunta: Tenure vs Monthly Charges")
    plt.savefig(output_path / "Scatter_Tenure_vs_Charges.png", dpi=300, bbox_inches='tight')
    plt.close()
def plot_num_services_count(df, output_path):
    """
    Volume di Churn in base al numero di servizi attivi.
    Insight: Più servizi = Maggiore 'stickiness' (meno churn).
    """
    services = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    df["NumServices"] = (df[services] == "Yes").sum(axis=1)
    plt.figure(figsize=(12, 7))
    ax = sns.countplot(x="NumServices", hue="ChurnValue", data=df, palette=PALETTE)
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 7), textcoords='offset points',
                        fontsize=10, fontweight='bold')
    plt.title("Volume di Utenti per Numero di Servizi e Status Churn")
    plt.savefig(output_path / "Count_Churn_by_NumServices.png", dpi=300, bbox_inches='tight')
    plt.close()
def plot_full_correlation_matrix(df, output_path):
    """
    Matrice di correlazione di Pearson su tutte le feature.
    Insight: Identifica quali variabili dummy (es. Contract_Two year) hanno l'impatto più forte.
    """
    cols_to_drop = ['CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code', 'Lat Long', 'Churn Reason']
    df_all = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    df_encoded = pd.get_dummies(df_all, drop_first=True)
    corr_matrix = df_encoded.corr(method='pearson')
    plt.figure(figsize=(20, 16))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', center=0,
                annot_kws={"size": 8}, linewidths=.5)
    plt.title("Matrice di Correlazione di Pearson (Dataset Completo)")
    plt.savefig(output_path / "full_correlation_matrix_pearson.png", dpi=300, bbox_inches='tight')
    plt.close()
def plot_churn_heatmap_grid(df, output_path):
    """
    Crea una heatmap che mostra la % di Churn per fasce di costo e anzianità.
    Ideale per il front-end: elimina l'overplotting e mostra la 'Danger Zone'.
    """
    # 1. Creiamo le classi di costo e anzianità (binning)
    df['Tenure_Bin'] = pd.cut(df['TenureMonths'], bins=10)
    df['Charges_Bin'] = pd.cut(df['MonthlyCharges'], bins=10)
    
    # 2. Calcoliamo la media del Churn (che è la probabilità) per ogni cella
    df['Churn_Numeric'] = df['ChurnValue'].astype(int)
    heatmap_data = df.pivot_table(index='Charges_Bin', columns='Tenure_Bin', 
                                  values='Churn_Numeric', aggfunc='mean')
    
    # Invertiamo l'asse Y per avere i costi alti in alto
    heatmap_data = heatmap_data.sort_index(ascending=False)
    # 3. Plotting
    plt.figure(figsize=(14, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".0%", cmap="YlOrRd", cbar_kws={'label': 'Probabilità di Churn'})
    
    plt.title("Mappa del Rischio: Costi Mensili vs Anzianità", fontsize=16, pad=20)
    plt.xlabel("Mesi di Anzianità (Tenure)", fontsize=12)
    plt.ylabel("Costi Mensili ($)", fontsize=12)
    
    plt.savefig(output_path / "Churn_Heatmap_Grid.png", dpi=300, bbox_inches='tight')
    plt.close()
def plot_demographic_analysis(df, output_path):
    """
    Analisi combinata Senior Citizen e Dependents.
    Insight: I Senior Citizen senza persone a carico sono spesso il segmento a più alto rischio.
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    # Senior Citizen vs Churn
    sns.countplot(x="SeniorCitizen", hue="ChurnValue", data=df, palette=PALETTE, ax=ax[0])
    ax[0].set_title("Senior Citizen Churn")
    # Dependents vs Churn
    sns.countplot(x="Dependents", hue="ChurnValue", data=df, palette=PALETTE, ax=ax[1])
    ax[1].set_title("Dependents vs Churn")
    plt.savefig(output_path / "demographic_churn.png", dpi=300, bbox_inches='tight')
    plt.close()
def plot_economic_value_dist(df, output_path):
    """
    Violin Plot: Distribuzione di Avg Monthly Spend per Churn.
    Insight: Mostra la 'pancia' della spesa dei clienti che se ne vanno rispetto a chi resta.
    """
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Churn Value", y="Avg Monthly Spend", data=df, palette=PALETTE, inner="quart")
    plt.title("Distribuzione Spesa Media Mensile per Status Churn")
    plt.savefig(output_path / "economic_value_distribution.png", dpi=300)
    plt.close()
def plot_charges_per_service_analysis(df, output_path):
    """
    KDE Plot del costo per singolo servizio.
    Insight: Se il costo per servizio supera una certa soglia, la probabilità di churn aumenta.
    """
    plt.figure(figsize=(10, 5))
    for val, label in [("0", "Stayed"), ("1", "Churned")]:
        sns.kdeplot(data=df[df["Churn Value"] == val], x="Charges per Service", 
                    fill=True, label=label, color=PALETTE[val], alpha=0.5)
    plt.title("Efficienza del Costo: Charges per Service vs Churn")
    plt.xlabel("Costo medio per singolo servizio attivo ($)")
    plt.legend()
    plt.savefig(output_path / "charges_per_service_kde.png", dpi=300, bbox_inches='tight')
    plt.close()
def plot_outliers_boxplots(df, output_path):
    """
    Boxplot mirato per le feature presenti nei tuoi dati raw.
    """
    # Selezioniamo solo le numeriche che possono avere outlier reali
    cols_to_plot = [
        'TenureMonths', 
        'MonthlyCharges', 
        'TotalCharges', 
        'Avg Monthly Spend', 
        'Charges per Service'
    ]
    # Assicuriamoci che TotalCharges sia numerico (spesso è object nei raw)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    plt.figure(figsize=(16, 10))
    for i, col in enumerate(cols_to_plot, 1):
        plt.subplot(2, 3, i)
        sns.boxplot(y=df[col], color="#e76f51", fliersize=5)
        plt.title(f'Distribuzione {col}')
        plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "boxplots_outliers.png", dpi=300)
    plt.close()
def plot_churn_heatmap_grid(df, output_path):
    """
    Crea una heatmap che mostra la % di Churn per fasce di costo e anzianità.
    Ideale per il front-end: elimina l'overplotting e mostra la 'Danger Zone'.
    """
    # 1. Creiamo le classi di costo e anzianità (binning)
    df['Tenure_Bin'] = pd.cut(df['TenureMonths'], bins=10)
    df['Charges_Bin'] = pd.cut(df['MonthlyCharges'], bins=10)
    # 2. Calcoliamo la media del Churn (che è la probabilità) per ogni cella
    # Assicuriamoci che Churn Value sia numerico
    df['Churn_Numeric'] = df['ChurnValue'].astype(int)
    heatmap_data = df.pivot_table(index='Charges_Bin', columns='Tenure_Bin',
                                  values='Churn_Numeric', aggfunc='mean')
    # Invertiamo l'asse Y per avere i costi alti in alto
    heatmap_data = heatmap_data.sort_index(ascending=False)
    # 3. Plotting
    plt.figure(figsize=(14, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".0%", cmap="YlOrRd", cbar_kws={'label': 'Probabilità di Churn'})
    plt.title("Mappa del Rischio: Costi Mensili vs Anzianità", fontsize=16, pad=20)
    plt.xlabel("Mesi di Anzianità (Tenure)", fontsize=12)
    plt.ylabel("Costi Mensili ($)", fontsize=12)
    plt.savefig(output_path / "Churn_Heatmap_Grid.png", dpi=300, bbox_inches='tight')
    plt.close()
# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Setup percorsi
    ROOT = Path(__file__).resolve().parents[1]
    PROC_DIR = ROOT / "data" / "processed"
    RAW_FILE = ROOT / "data" / "raw" / "Telco_customer_churn.csv"
    OUT_DIR = ROOT / "outputs" / "plots"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Base analisi post-elaborazione (train+test processed)
    df_train = pd.read_csv(PROC_DIR / "train_raw.csv")
    df_test = pd.read_csv(PROC_DIR / "test_raw.csv")
    df = harmonize_columns(pd.concat([df_train, df_test], axis=0, ignore_index=True))
    # Dati raw solo per mappa geografica (Latitude/Longitude)
    raw_df = harmonize_columns(pd.read_csv(RAW_FILE))
    # Chiamata a tutte le funzioni
    plot_churn_distribution(df, OUT_DIR)
    plot_contract_churn(df, OUT_DIR)
    plot_tenure_group_rate(df, OUT_DIR)
    plot_scatter_tenure_charges(df, OUT_DIR)
    plot_churn_heatmap_grid(df, OUT_DIR)
    plot_tenure_kde(df, OUT_DIR)
    plot_monthly_charges_kde(df, OUT_DIR)
    plot_services_and_payment(df, OUT_DIR)
    plot_map_distribution(raw_df, OUT_DIR)
    plot_num_services_count(df, OUT_DIR)
    plot_full_correlation_matrix(df, OUT_DIR)
    plot_demographic_analysis(df, OUT_DIR)
    plot_economic_value_dist(df, OUT_DIR)
    plot_charges_per_service_analysis(df, OUT_DIR)
    plot_outliers_boxplots(df, OUT_DIR)
    export_chart_comments(OUT_DIR)
    print(f"Analisi completata. Grafici salvati in: {OUT_DIR}")