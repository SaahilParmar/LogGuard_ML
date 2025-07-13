✅ All Commands You Can Run


RUN ALL TEST - 

PYTHONPATH=. pytest

✅ 1. Run the Main Pipeline

→ Run your log processing pipeline without ML anomaly detection:

python main.py --logfile data/sample_log.log

✅ Result:

parses logs

generates HTML report



---

→ Run pipeline with ML anomaly detection:

python main.py --logfile data/sample_log.log --ml

✅ Result:

detects anomalies

flags suspicious logs

generates HTML report



---

✅ 2. Run All Tests

→ Run all tests in your repo:

pytest

Or specifically your file:

pytest tests/test_log_parser.py

✅ Output:

tests/test_log_parser.py ... [100%]

✅ Tells you if your parser and ML logic work.


---

✅ 3. Check Installed Packages

→ Check all installed dependencies:

pip list


---

✅ 4. Check Specific Package Version

e.g. check pandas version:

pip show pandas

or in Python:

import pandas
print(pandas.__version__)


---

✅ 5. Generate Allure Reports (if you integrate Allure)

If you’ve run pytest with Allure:

pytest --alluredir=reports/allure-results

→ Then generate the HTML report:

allure generate reports/allure-results -o reports/html --clean

→ Open the report:

allure open reports/html

✅ This step is optional if you’re using Allure for reporting.


---

✅ 6. Remove Old Virtual Env (If You Switch)

If you ever switch from venv/ to .venv/, remove the old one:

rm -rf venv

✅ Not required now if you’re already using .venv.


---

✅ 7. Activate / Deactivate Virtual Environment

Activate:

source .venv/bin/activate

Deactivate:

deactivate


---

✅ Quick Reference

Task	Command

Run pipeline (no ML)	python main.py --logfile data/sample_log.log
Run pipeline with ML	python main.py --logfile data/sample_log.log --ml
Run all tests	pytest
Run specific test file	pytest tests/test_log_parser.py
Check installed packages	pip list
Show package details	pip show pandas
Generate Allure report	allure generate ...
Open Allure report	allure open ...














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