# LogGuard ML

🚀 **LogGuard ML** is a Python-based log analysis framework combining:
- Regex log parsing
- Machine Learning anomaly detection
- Beautiful HTML reports

---

## 📂 Project Structure

logguard_ml/ ├── config/ │   └── config.yaml ├── data/ │   └── sample_log.log ├── reports/ │   └── (HTML reports generated here) ├── utils/ │   ├── log_parser.py │   └── ml_model.py ├── tests/ │   └── test_log_parser.py ├── images/ │   └── (screenshots for README) ├── requirements.txt ├── README.md ├── troubleshooting_log.md └── main.py

---

## ⚙️ How to Install

```bash
pip install -r requirements.txt


---

🚀 How to Run

Basic Log Parsing

python main.py --logfile data/sample_log.log

With ML Anomaly Detection

python main.py --logfile data/sample_log.log --ml

Output

Generates:

reports/anomaly_report.html


---

📊 Example Screenshot

(Replace this with your screenshot images)




---

✅ Features

Regex log parsing from YAML config

Isolation Forest ML for anomalies

HTML reporting with Plotly charts

Fully tested with pytest



---