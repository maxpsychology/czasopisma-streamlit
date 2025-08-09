
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Wykaz czasopism — wyszukiwarka", layout="wide")

st.title("🔎 Wyszukiwarka czasopism (lista MEiN 2024) — wersja z Excelem")

st.markdown("""
Ta wersja **czyta bezpośrednio plik Excel (.xlsx)** w układzie jak w MEiN 2024.
Możesz skorzystać z jednej z opcji:
1) **Plik lokalny w repo** – domyślnie: `wykaz_mein_2024.xlsx` w katalogu aplikacji.  
2) **Wgraj własny plik** (xlsx lub csv).  
3) **Podaj adres URL** do pliku (np. RAW z GitHuba).
""")

@st.cache_data
def read_mein_excel(file_like):
    # Dane MEiN mają nagłówek w wierszu 2 (indeks 1), a dane od wiersza 3 (indeks 2).
    raw = pd.read_excel(file_like, sheet_name="Czasopisma _nauk", header=None)
    header = raw.iloc[1].tolist()
    df = raw.iloc[2:].copy()
    df.columns = header

    # Zidentyfikuj kolumny dyscyplin: wszystko za 'Punktacja'
    start_idx = df.columns.get_loc('Punktacja') + 1
    names_row = raw.iloc[0, start_idx:].tolist()  # nazwy dyscyplin
    codes_row = raw.iloc[1, start_idx:].tolist()  # kody (np. 511)
    disc_cols = df.columns[start_idx:]
    code_to_name = {codes_row[i]: names_row[i] for i in range(len(disc_cols))}

    def extract_disciplines(row):
        out = []
        for code in disc_cols:
            if str(row.get(code, "")).strip().lower() == 'x':
                out.append(code_to_name.get(code, str(code)))
        return ", ".join(out)

    df['Dyscypliny'] = df.apply(extract_disciplines, axis=1)

    # Zachowaj kluczowe kolumny i porządki
    keep = ["Tytuł 1","Tytuł 2","issn","e-issn","Punktacja","Dyscypliny"]
    df = df[keep].copy()

    # Poradź sobie z potencjalnymi duplikatami nazw kolumn w źródle
    from collections import Counter
    def dedup_columns(cols):
        counts = Counter()
        new_cols = []
        for c in cols:
            counts[c] += 1
            if counts[c] == 1:
                new_cols.append(c)
            else:
                new_cols.append(f"{c}.{counts[c]-1}")
        return new_cols
    df.columns = dedup_columns(df.columns)

    # Typy i czyszczenie
    df["Punktacja"] = pd.to_numeric(df["Punktacja"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["Tytuł 1", "Punktacja"])
    return df

@st.cache_data
def read_any(file):
    # Wspiera: .xlsx MEiN, .csv oraz URL (https/RAW z GitHuba)
    if isinstance(file, str) and file.startswith(("http://","https://")):
        # URL (np. RAW GitHub) — pandas obsłuży
        if file.lower().endswith(".csv"):
            return pd.read_csv(file)
        # xlsx
        return read_mein_excel(file)

    # Lokalny upload (UploadedFile)
    name = getattr(file, "name", "")
    if name.endswith(".csv"):
        return pd.read_csv(file)
    if name.endswith(".xlsx"):
        return read_mein_excel(file)

    # Ścieżka lokalna
    if isinstance(file, str):
        if file.lower().endswith(".csv"):
            return pd.read_csv(file)
        if file.lower().endswith(".xlsx"):
            return read_mein_excel(file)

    raise ValueError("Nieobsługiwany format lub brak pliku.")

# Panel boczny: wybór źródła
with st.sidebar:
    st.header("Źródło danych")
    source = st.radio("Wybierz:", ["Plik w repo (xlsx)", "Wgraj plik", "URL"])

    if source == "Plik w repo (xlsx)":
        st.caption("Upewnij się, że plik `wykaz_mein_2024.xlsx` jest w tym samym katalogu co `app.py`.")
        path = st.text_input("Nazwa/ścieżka pliku", value="wykaz_mein_2024.xlsx")
        df = read_any(path)

    elif source == "Wgraj plik":
        up = st.file_uploader("Wgraj .xlsx (MEiN) lub .csv", type=["xlsx","csv"])
        if up is None:
            st.stop()
        df = read_any(up)

    else:  # URL
        url = st.text_input("Podaj URL (np. RAW z GitHuba)", placeholder="https://raw.githubusercontent.com/user/repo/branch/plik.xlsx")
        if not url:
            st.stop()
        df = read_any(url)

# Przygotowanie listy dyscyplin
def split_disciplines(s):
    if pd.isna(s):
        return []
    return [x.strip() for x in str(s).split(",") if x and str(x).strip()]

all_disciplines = sorted({d for row in df["Dyscypliny"].apply(split_disciplines) for d in row})

# Filtry
with st.sidebar:
    st.header("Filtry")
    title_query = st.text_input("Szukaj w tytule", placeholder="np. psychology, management...")
    st.markdown("**Dyscypliny**")
    selected = []
    cols = st.columns(2)
    for i, d in enumerate(all_disciplines):
        if cols[i % 2].checkbox(d, value=False, key=f"disc_{i}"):
            selected.append(d)

    # Suwak punktacji
    min_p = int(pd.to_numeric(df["Punktacja"], errors="coerce").dropna().min())
    max_p = int(pd.to_numeric(df["Punktacja"], errors="coerce").dropna().max())
    step = 10 if (max_p - min_p) >= 10 else 1
    points = st.slider("Punktacja (zakres)", min_value=min_p, max_value=max_p, value=(min_p, max_p), step=step)

# Filtrowanie
f = df.copy()

if title_query:
    q = title_query.lower()
    mask = f["Tytuł 1"].astype(str).str.lower().str.contains(q, na=False)
    if "Tytuł 2" in f.columns:
        mask |= f["Tytuł 2"].astype(str).str.lower().str.contains(q, na=False)
    f = f[mask]

if selected:
    f = f[f["Dyscypliny"].apply(lambda s: any(d in s for d in selected))]

f = f[(f["Punktacja"] >= points[0]) & (f["Punktacja"] <= points[1])]

st.success(f"Znaleziono {len(f):,} pozycji")

st.dataframe(f, use_container_width=True, height=600)

# Pobieranie wyników
csv = f.to_csv(index=False).encode("utf-8-sig")
st.download_button("📥 Pobierz wyniki (CSV)", data=csv, file_name="wyniki_czasopisma.csv", mime="text/csv")

st.markdown("---")
st.caption("Uwaga: jeśli MEiN zmieni układ arkusza, trzeba będzie zaktualizować parser w aplikacji.")
