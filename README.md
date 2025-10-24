# applied-ml-foster

> Start a professional Python project.
>
> Clone repo to local
> Install Extensions
> Set up virtual environment
> Git Commit
> Git pull origin main
> Run Tests: 
uv sync --extra dev --extra docs --upgrade
uv cache clean
git add .
uvx ruff check --fix
uvx pre-commit autoupdate
uv run pre-commit run --all-files
git add .
uv run pytest

🧭 Data Cleaning Workflow

The cleaning process follows these steps:

1. Load the Dataset

Reads the CSV file into a pandas DataFrame and prints the initial shape and missing values.

df = pd.read_csv(DATA_PATH)

2. Identify Missing Values

Replaces placeholder missing values like ? or empty strings ("") with proper NaN.

df.replace("?", pd.NA, inplace=True)
df.replace("", pd.NA, inplace=True)

3. Convert Numeric Columns

Ensures numeric columns are stored as numbers (not strings).
Invalid entries are automatically converted to NaN.

df[col] = pd.to_numeric(df[col], errors="coerce")

4. Remove Rows Missing the Target Variable

Rows missing the price value are dropped, since price is the key target variable.

df = df.dropna(subset=["price"])

5. Fill Missing Numeric Values

Missing numeric entries are replaced with the median of that column.

df[col].fillna(df[col].median(), inplace=True)

6. Fill Missing Categorical Values

Missing text/categorical entries are replaced with the most frequent (mode) value.

df[col].fillna(df[col].mode()[0], inplace=True)

7. Standardize Text Columns

All text columns are cleaned to lowercase and stripped of whitespace for consistency.

df[col] = df[col].astype(str).str.lower().str.strip()

8. Convert Text Numbers to Numeric

Creates a numeric version of the “doors” column for analysis or modeling.

df["doors_num"] = df["doors"].map({"two": 2, "three": 3, "four": 4, "five": 5})

9. Save the Cleaned Data

Saves the fully cleaned dataset as a new CSV file.

df.to_csv(OUTPUT_PATH, index=False)

✅ Output

When the script completes successfully, you’ll see logs like:

Reading data from: automobile_price_data3.csv
Initial shape: (205, 11)
Dropped 4 rows missing price. New shape: (201, 11)
Filled numeric column 'bhp' with median = 110.0
✅ Cleaned data saved to: automobile_price_data3_cleaned.csv


Your cleaned dataset is now ready for exploratory data analysis (EDA) or machine learning.

# 🤖 Step 2: Train–Test Split

Once the data is cleaned, the next step is to **prepare it for machine learning** by splitting it into a training set and a test set.

---

## ⚙️ How It Works

The dataset is divided into:
- **Training set (80%)** — used to train the model  
- **Test set (20%)** — held out for final evaluation  

This ensures that model performance is measured on unseen data.

---

## 🧩 Script Overview: `2_train_test_split.py`

### 1. **Load Cleaned Data**
```python
df = pd.read_csv("automobile_price_data3_cleaned.csv")
2. Define Features and Target
The model predicts price based on all other columns.

python
Copy code
X = df.drop(columns=["price"])
y = df["price"]
3. Split Data
Using scikit-learn’s train_test_split:

python
Copy code
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
4. Save Output
Two new CSV files are created for easy inspection:

File	Purpose
train_data.csv	Training dataset (80%)
test_data.csv	Test dataset (20%)

✅ Output Example
kotlin
Copy code
✅ Data Split Complete
Training set: 161 rows
Test set: 40 rows
Training data saved to: train_data.csv
Test data saved to: test_data.csv

- Additional instructions: See the [pro-analytics-02](https://denisecase.github.io/pro-analytics-02/) guide.
- Project organization: [STRUCTURE](./STRUCTURE.md)
- Build professional skills:
  - **Environment Management**: Every project in isolation
  - **Code Quality**: Automated checks for fewer bugs
  - **Documentation**: Use modern project documentation tools
  - **Testing**: Prove your code works
  - **Version Control**: Collaborate professionally

---

## About this Repository

Starter files for the example labs:

- notebooks/example01 folder
- notebooks/example02 folder

## Folders for Projects

Each project will be completed in its own folder.

- notebooks/project01 folder:
  - ml01.ipynb - COMPLETE THIS
  - ml01.py - working script with just the code
  - README.md - instructions - modify this to present your lab project

---

## WORKFLOW 1. Set Up Machine

Proper setup is critical.
Complete each step in the following guide and verify carefully.

- [SET UP MACHINE](./SET_UP_MACHINE.md)

---

## WORKFLOW 2. Set Up Project

After verifying your machine is set up, set up a new Python project by copying this template.
Complete each step in the following guide.

- [SET UP PROJECT](./SET_UP_PROJECT.md)

It includes the critical commands to set up your local environment (and activate it):

```shell
uv venv
uv python pin 3.12
uv sync --extra dev --extra docs --upgrade
uv run pre-commit install
uv run python --version
```

**Windows (PowerShell):**

```shell
.\.venv\Scripts\activate
```

**macOS / Linux / WSL:**

```shell
source .venv/bin/activate
```

---

## WORKFLOW 3. Daily Workflow

Please ensure that the prior steps have been verified before continuing.
When working on a project, we open just that project in VS Code.

### 3.1 Git Pull from GitHub

Always start with `git pull` to check for any changes made to the GitHub repo.

```shell
git pull
```

### 3.2 Run Checks as You Work

This mirrors real work where we typically:

1. Update dependencies (for security and compatibility).
2. Clean unused cached packages to free space.
3. Use `git add .` to stage all changes.
4. Run ruff and fix minor issues.
5. Update pre-commit periodically.
6. Run pre-commit quality checks on all code files (**twice if needed**, the first pass may fix things).
7. Run tests.

In VS Code, open your repository, then open a terminal (Terminal / New Terminal) and run the following commands one at a time to check the code.

```shell
git pull
uv sync --extra dev --extra docs --upgrade
uv cache clean
git add .
uvx ruff check --fix
uvx pre-commit autoupdate
uv run pre-commit run --all-files
git add .
uv run pytest
```

NOTE: The second `git add .` ensures any automatic fixes made by Ruff or pre-commit are included before testing or committing.
Running `uv run pre-commit run --all-files` twice may be helpful if the first time doesn't pass. 

<details>
<summary>Click to see a note on best practices</summary>

`uvx` runs the latest version of a tool in an isolated cache, outside the virtual environment.
This keeps the project light and simple, but behavior can change when the tool updates.
For fully reproducible results, or when you need to use the local `.venv`, use `uv run` instead.

</details>

### 3.3 Build Project Documentation

Make sure you have current doc dependencies, then build your docs, fix any errors, and serve them locally to test.

```shell
uv run mkdocs build --strict
uv run mkdocs serve
```

- After running the serve command, the local URL of the docs will be provided. To open the site, press **CTRL and click** the provided link (at the same time) to view the documentation. On a Mac, use **CMD and click**.
- Press **CTRL c** (at the same time) to stop the hosting process.

### 3.4 Execute

This project includes demo code.
Run the demo Python modules to confirm everything is working.

In VS Code terminal, run:

```shell
uv run python notebooks/project01/ml01.py
```

A new window with charts should appear. Close the window to finish the execution. 
If this works, your project is ready! If not, check:

- Are you in the right folder? (All terminal commands are to be run from the root project folder.)
- Did you run the full `uv sync --extra dev --extra docs --upgrade` command?
- Are there any error messages? (ask for help with the exact error)

## Update this README as you work

Add commands to run additional scripts as you work through the course (update the path and file name as needed).

---

### 3.5 Git add-commit-push to GitHub

Anytime we make working changes to code is a good time to git add-commit-push to GitHub.

1. Stage your changes with git add.
2. Commit your changes with a useful message in quotes.
3. Push your work to GitHub.

```shell
git add .
git commit -m "describe your change in quotes"
git push -u origin main
```

This will trigger the GitHub Actions workflow and publish your documentation via GitHub Pages.

### 3.6 Modify and Debug

With a working version safe in GitHub, start making changes to the code.

Before starting a new session, remember to do a `git pull` and keep your tools updated.

Each time forward progress is made, remember to git add-commit-push.
