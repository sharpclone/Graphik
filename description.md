# Graphik

Lokale Streamlit-App zur Erstellung von Protokoll-gerechten linearen Diagrammen mit:
- Messpunkten und vertikalen Fehlerbalken
- optionaler Verbindungslinie zwischen Messpunkten
- roter Ausgleichsgerade (`y = kx + l`)
- grünen Fehlergeraden (max/min Steigung, centroid-basiert)
- optionalen Steigungsdreiecken (`Δx`, `Δy`)
- Export nach PNG, SVG, PDF und Kurzprotokoll (MD/TXT)

## Projektstruktur

```text
.
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── sample_measurements.csv
├── src/
│   ├── __init__.py
│   ├── calculations.py
│   ├── config.py
│   ├── data_io.py
│   ├── export_utils.py
│   ├── geometry.py
│   ├── plotting.py
│   ├── ui_helpers.py
│   └── ui_state.py
└── tests/
    ├── test_calculations.py
    └── test_data_io.py
```

## Installation und Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Die App ist auf **lokalen Betrieb** konfiguriert und bindet standardmaessig auf `127.0.0.1:8501` (`.streamlit/config.toml`).

## Kernfunktionen

1. **Dateninput**
- Manuelle Tabellenbearbeitung (`st.data_editor`)
- CSV-, Excel- oder OpenDocument-Upload (`.csv`, `.xlsx`, `.xls`, `.ods`)
- CSV-Import erkennt typische Excel/LibreOffice-Formate automatisch (`,` oder `;`, Punkt- oder Komma-Dezimaltrennzeichen, gaengige Encodings)
- Einfache Spaltenzuordnung im UI: nur `x`, `y` und Fehler-Spalte waehlen
- Wenn eine Tabelle nur `Unnamed:*`-Spalten hat (z. B. bei ODS-Headern mit Formeln), werden die ersten Spalten automatisch als `m`, `T_mean`, `sigma_T`, `y`, `sigma_y` benannt.
- Header-Erkennung scannt auch die ersten Zeilen und kann Headerzeilen nach leeren Zeilen automatisch uebernehmen (typisch bei ODS/Excel-Layouts).
- Unterstützte Spalten:
  - `x` (z. B. `m`)
  - `y`
  - optional `sigma_y`

2. **Plot (Laborstil)**
- Schwarze Messpunkte mit vertikalen Fehlerbalken
- Optional: Verbindungslinie durch Messpunkte
- Editierbare Achsenlabels (Default: `m [g]`, `T² [s²]`)
- Umschaltbare y-Skalierung: `linear` oder `log10 (semilog)`
- Ein-/zweidekadische Darstellung fuer log-y (`auto`, `1 decade`, `2 decades`)
- Optional: MathJax/LaTeX-Schreibweise in Plotly-Labels (z. B. `$T^2\\,[s^2]$`, `$\\Delta m$`)
- Optionale Gitterlinien
- Optionale manuelle Achsenbereiche

3. **Ausgleichsgerade**
- Lineare Regression mit SciPy
- Optional exponenteilles Modell fuer semilog Messungen:
  - `ln(y) = a*x + ln(k)`  <=>  `y = k*exp(a*x)`
- Darstellung als rote Linie
- Ausgabe von `k_fit`, `l_fit`, Gleichung
- Steigungsdreieck mit automatischen oder benutzerdefinierten Punkten A/B

4. **Fehlergeraden**
- Centroid-Methode (durch Schwerpunkt `(x̄, ȳ)`)
- Berechnung von `k_max`, `k_min`, `delta_k = (k_max-k_min)/2`
- Kompatibilitätsbedingung je Messpunkt:
  - Linienwert bei `x_i` muss in `[y_i - sigma_y_i, y_i + sigma_y_i]` liegen
- Darstellung als grüne Linien (gestrichelt/gesprenkelt)
- Optional nur eine extreme Linie (`k_max` oder `k_min`)
- Optionale Steigungsdreiecke auch für Fehlergeraden

5. **Wissenschaftliches Output-Panel**
- `y = kx + l`
- `k_fit`, `l_fit`
- `k_triangle`
- `k_max`, `k_min`, `delta_k`
- Empfohlene Endangabe: `k = k_fit ± delta_k`

6. **Export**
- Plot als PNG/SVG/PDF (via Plotly + Kaleido)
- PNG-Export mit einstellbarer Papiergroesse (A4/A5/Letter), Orientierung (Portrait/Landscape) und DPI
- Analyse-CSV
- Kurzbericht als Markdown/TXT mit Daten und Kennwerten

## Fehlergeraden-Algorithmus (Kurzkommentar)

Um subjektive Fehlergeraden nach üblichem Laborleitfaden konsistent zu erzeugen:

1. Schwerpunkt berechnen: `x̄ = mean(x_i)`, `ȳ = mean(y_i)`.
2. Alle Fehlergeraden werden auf `y = k(x - x̄) + ȳ` eingeschränkt.
3. Für jeden Punkt `(x_i, y_i ± sigma_y_i)` ergibt sich ein zulässiges Intervall für `k`.
4. Schnitt aller Intervalle liefert `[k_min, k_max]`.
5. Daraus:
   - `l_min = ȳ - k_min * x̄`
   - `l_max = ȳ - k_max * x̄`
   - `delta_k = (k_max - k_min)/2`

Wenn der Intervallschnitt leer ist, existiert keine centroid-kompatible Fehlergerade (die App zeigt dann eine Warnung).

## Tests

```bash
pytest -q
```

Abgedeckt:
- lineare Regression
- Kompatibilität von Linien mit Fehlerbalken
- `k_max`, `k_min`, `delta_k`

## Beispiel-Workflow (Screenshot-orientiert)

1. App starten: `streamlit run app.py`
2. In der Sidebar auf **Use sample data** klicken.
3. Sicherstellen:
- `x column = m`
- `y column = y` oder eine andere vorab berechnete Messgroesse
- `error column = sigma_y` oder eine andere vorab berechnete Unsicherheitsspalte
4. **Show Ausgleichsgerade** und **Show Fehlergeraden** aktivieren.
5. Optional **Show fit slope triangle** aktivieren.
6. Ergebnisansicht enthält dann:
- schwarze Punkte + Fehlerbalken
- rote Ausgleichsgerade
- grüne Fehlergeraden
- `Δx`, `Δy`-Labels am Dreieck
- Kennwerte im rechten Scientific-output-Panel
7. Export über **Download PNG**, **Download SVG** oder **Download PDF**.

## Semilog-Workflow (Absorberdicke vs. Intensitaet)

1. Tabelle aus Excel/ODS hochladen (`Dicke`, `Imp`, `err`).
2. **Mapping mode = Simple** setzen:
- `x column = Dicke`
- `y column = Imp`
- `error column = err`
3. In **Plot**:
- `y-axis scale = log10 (semilog paper)`
- optional `Log paper decades = 1 decade` oder `2 decades`
4. Achsenlabels setzen, z. B.:
- `x: d [10^-6 m]`
- `y: I [1/s]`
5. Plot zeigt x linear, y logarithmisch und Fehlerbalken; Ausreisser sind dadurch direkt sichtbar.
