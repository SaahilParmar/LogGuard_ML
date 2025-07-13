---

👨‍💻 Author

Saahil Parmar

---

# ✅ **3. troubleshooting_log.md**

Save as:

logguard_ml/troubleshooting_log.md

✅ **Full Code:**

```markdown
# Troubleshooting Log - LogGuard ML

✅ **Documenting real issues encountered and solutions.**

---

## 1. Allure Not Installed

**Error:**

bash: allure: command not found

**Solution:**

- Download Allure binary manually
- Add to PATH:
    ```
    export PATH=$PATH:/opt/allure/bin
    source ~/.bashrc
    ```

---

## 2. Chrome Binary Missing

**Error:**

selenium.common.exceptions.WebDriverException: Message: unknown error: cannot find Chrome binary

**Solution:**

Install Google Chrome:

```bash
sudo apt install -y google-chrome-stable


---

3. No Such Element Exception

Error:

selenium.common.exceptions.NoSuchElementException: Message: no such element

Solution:

Check selector

Add wait conditions

Confirm page loads properly



---

4. YAML Parse Error

Error:

yaml.YAMLError: while scanning a simple key

Solution:

Check YAML syntax:

Colons require space after them

Indentation is crucial



---
