from click import prompt
from flask import Flask, render_template, request
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.pdfgen import canvas
from flask import send_file
import google.generativeai as genai

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Regression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

app = Flask(__name__)



genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel(
    "models/gemini-2.5-flash"
)
latest_report = {}
latest_df = None

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists("static"):
    os.makedirs("static")

ALLOWED_EXTENSIONS = {"csv", "xlsx", "json"}


def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        if "file" not in request.files:
            return "No file selected"

        file = request.files["file"]

        if file.filename == "":
            return "Please select a file"

        if file and allowed_file(file.filename):

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                file.filename
            )

            file.save(filepath)

            try:

                # READ FILE

                if file.filename.endswith(".csv"):
                    df = pd.read_csv(filepath)

                elif file.filename.endswith(".xlsx"):
                    df = pd.read_excel(filepath)

                elif file.filename.endswith(".json"):
                    df = pd.read_json(filepath)

                else:
                    return "Unsupported File"
                global latest_df
                latest_df = df

                # BASIC STATS

                rows = df.shape[0]
                cols = df.shape[1]

                missing = int(df.isnull().sum().sum())
                duplicates = int(df.duplicated().sum())

                numeric_cols = len(
                    df.select_dtypes(include="number").columns
                )

                categorical_cols = len(
                    df.select_dtypes(include="object").columns
                )

                # TABLE PREVIEW

                table = df.head(10).to_html(
                    classes="table table-dark table-striped table-hover",
                    index=False
                )

                summary_table = df.describe().to_html(
                classes="table table-dark table-striped",
                border=0
                )

                datatype_table = pd.DataFrame(
                df.dtypes,
                columns=["Data Type"]
                ).to_html(
    classes="table table-dark table-striped",
    border=0
)
                # DATASET TYPE

                dataset_type = "General Dataset"

                if "LoanAmount" in df.columns:
                    dataset_type = "Loan Prediction Dataset"

                elif "Salary" in df.columns:
                    dataset_type = "Employee Salary Dataset"

                elif "Attendance" in df.columns:
                    dataset_type = "Student Performance Dataset"

                elif "Sales" in df.columns:
                    dataset_type = "Sales Dataset"

                # HEATMAP

                heatmap_available = False

                numeric_df = df.select_dtypes(include="number")

                if numeric_df.shape[1] > 1:

                    plt.figure(figsize=(10, 6))

                    sns.heatmap(
                        numeric_df.corr(),
                        annot=True,
                        cmap="coolwarm"
                    )

                    plt.tight_layout()

                    plt.savefig(
                        "static/heatmap.png"
                    )

                    plt.close()

                    heatmap_available = True

                    if len(numeric_df.columns) > 0:

                        plt.figure(figsize=(8,5))

                        numeric_df.iloc[:,0].hist()

                        plt.title(
                        f"Histogram - {numeric_df.columns[0]}"
    )

                        plt.tight_layout()

                        plt.savefig(
                        "static/histogram.png"
    )

                        plt.close()

                    # ==========================
# BOX PLOT
# ==========================

                if len(numeric_df.columns) > 0:

                    plt.figure(figsize=(8,5))

                    sns.boxplot(
                    x=numeric_df.iloc[:,0]
    )

                    plt.title(
                    f"Box Plot - {numeric_df.columns[0]}"
    )

                    plt.tight_layout()

                    plt.savefig(
                    "static/boxplot.png"
    )

                    plt.close()


# ==========================
# SCATTER PLOT
# ==========================

                if len(numeric_df.columns) >= 2:

                    plt.figure(figsize=(8,5))

                    plt.scatter(
                    numeric_df.iloc[:,0],
                    numeric_df.iloc[:,1]
    )

                    plt.xlabel(
                    numeric_df.columns[0]
    )

                    plt.ylabel(
                        numeric_df.columns[1]
                    )

                    plt.title(
                        "Scatter Plot"
                    )

                    plt.tight_layout()

                    plt.savefig(
                        "static/scatter.png"
                    )

                    plt.close()


# ==========================
# PIE CHART
# ==========================

                if len(df.columns) > 0:

                    first_col = df.columns[0]

                    pie_data = (
                    df[first_col]
                    .astype(str)
                    .value_counts()
                    .head(5)
                )

                    plt.figure(figsize=(6,6))

                    plt.pie(
                        pie_data,
                        labels=pie_data.index,
                        autopct="%1.1f%%"
                    )

                    plt.title(
                        f"Pie Chart - {first_col}"
                    )

                    plt.savefig(
                        "static/piechart.png"
                    )

                    plt.close()

                # AI INSIGHTS

                insights = [
                    f"Dataset contains {rows} rows.",
                    f"Dataset contains {cols} columns.",
                    f"{missing} missing values detected.",
                    f"{duplicates} duplicate rows detected.",
                    f"{numeric_cols} numeric columns found.",
                    f"{categorical_cols} categorical columns found."
                ]
                ai_summary = "AI Summary Not Available"

                try:

                    sample_data = df.head(10).to_string()

                    prompt = f"""
                    Analyze this dataset sample and provide:

                    1. Dataset purpose
                    2. Important patterns
                    3. Suitable ML problem
                    4. Business insights

                    Dataset:

                    {sample_data}
                    """

                    model = genai.GenerativeModel(
                    "models/gemini-2.5-flash"
    )

                    response = model.generate_content(
                    prompt
    )

                    ai_summary = response.text

                except Exception as e:

                    ai_summary = f"Gemini Error: {str(e)}"

                # MACHINE LEARNING

                accuracies = {}
                best_model = "Not Available"

                try:

                    target_column = df.columns[-1]

                    insights.append(
                        f"Target Column Detected: {target_column}"
                    )

                    temp_df = df.copy()

                    temp_df.fillna(0, inplace=True)

                    for col in temp_df.columns:

                        if temp_df[col].dtype == "object":

                            encoder = LabelEncoder()

                            temp_df[col] = encoder.fit_transform(
                                temp_df[col].astype(str)
                            )

                    X = temp_df.drop(
                        target_column,
                        axis=1
                    )

                    y = temp_df[target_column]

                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=0.2,
                        random_state=42
                    )

                    # CLASSIFICATION

                    if y.nunique() <= 20:

                        insights.append(
                            "Dataset Type: Classification"
                        )

                        models = {

                            "Logistic Regression":
                            LogisticRegression(max_iter=1000),

                            "Decision Tree":
                            DecisionTreeClassifier(),

                            "Random Forest":
                            RandomForestClassifier(),

                            "KNN":
                            KNeighborsClassifier(),

                            "SVM":
                            SVC(),

                            "Naive Bayes":
                            GaussianNB()

                        }

                        for name, model in models.items():

                            model.fit(
                                X_train,
                                y_train
                            )

                            pred = model.predict(
                                X_test
                            )

                            score = round(
                                accuracy_score(
                                    y_test,
                                    pred
                                ) * 100,
                                2
                            )

                            accuracies[name] = score

                    # REGRESSION

                    else:

                        insights.append(
                            "Dataset Type: Regression"
                        )

                        models = {

                            "Linear Regression":
                            LinearRegression(),

                            "Decision Tree Regressor":
                            DecisionTreeRegressor(),

                            "Random Forest Regressor":
                            RandomForestRegressor()

                        }

                        for name, model in models.items():

                            model.fit(
                                X_train,
                                y_train
                            )

                            pred = model.predict(
                                X_test
                            )

                            score = round(
                                r2_score(
                                    y_test,
                                    pred
                                ) * 100,
                                2
                            )

                            accuracies[name] = score

                    if len(accuracies) > 0:

                        best_model = max(
                            accuracies,
                            key=accuracies.get
                        )

                        insights.append(
                            f"Best Model: {best_model}"
                        )

                except Exception as ml_error:

                    insights.append(
                        f"ML Error: {str(ml_error)}"
                    )
                global latest_report

                latest_report = {
                "rows": rows,
                "cols": cols,
                "missing": missing,
                "duplicates": duplicates,
                "dataset_type": dataset_type,
                "best_model": best_model
}
                return render_template(
                    "dashboard.html",
                    rows=rows,
                    cols=cols,
                    missing=missing,
                    duplicates=duplicates,
                    numeric_cols=numeric_cols,
                    categorical_cols=categorical_cols,
                    dataset_type=dataset_type,
                    table=table,
                    insights=insights,
                    heatmap_available=heatmap_available,
                    accuracies=accuracies,
                    best_model=best_model,  
                    summary_table=summary_table,
                    datatype_table=datatype_table,
                    ai_summary=ai_summary,
                )

            except Exception as e:

                return f"Error Reading Dataset: {str(e)}"

        return "Only CSV, Excel and JSON files are allowed"
        # ==========================
# MODEL COMPARISON CHART
# ==========================

        if len(accuracies) > 0:

                    plt.figure(figsize=(10,5))

                    plt.bar(
                    accuracies.keys(),
                    accuracies.values()
                )

                    plt.title(
                        "Model Performance Comparison"
                    )

                    plt.ylabel("Score (%)")

                    plt.xticks(rotation=20)

                    plt.tight_layout()

                    plt.savefig(
                        "static/model_chart.png"
                    )

                    plt.close()

    return render_template("index.html")

@app.route("/download-report")
def download_report():

    pdf_path = "report/report.pdf"

    if not os.path.exists("report"):
        os.makedirs("report")

    pdf = canvas.Canvas(pdf_path)

    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(180, 800, "DataMind AI Report")

    pdf.setFont("Helvetica", 12)

    pdf.drawString(
        50, 740,
        f"Rows: {latest_report.get('rows')}"
    )

    pdf.drawString(
        50, 720,
        f"Columns: {latest_report.get('cols')}"
    )

    pdf.drawString(
        50, 700,
        f"Missing Values: {latest_report.get('missing')}"
    )

    pdf.drawString(
        50, 680,
        f"Duplicates: {latest_report.get('duplicates')}"
    )

    pdf.drawString(
        50, 660,
        f"Dataset Type: {latest_report.get('dataset_type')}"
    )

    pdf.drawString(
        50, 640,
        f"Best Model: {latest_report.get('best_model')}"
    )

    pdf.save()

    return send_file(
        pdf_path,
        as_attachment=True
    )
@app.route("/chat")


def chat():

        global latest_df

        if latest_df is None:
            return "No dataset uploaded"

        question = request.args.get("q")

        if not question:
            return "Please ask a question"

        question = question.lower()

        if "rows" in question:
            return f"Dataset contains {latest_df.shape[0]} rows"

        elif "columns" in question:
            return f"Dataset contains {latest_df.shape[1]} columns"

        elif "missing" in question:
            return str(latest_df.isnull().sum())

        elif "average" in question:

            numeric_df = latest_df.select_dtypes(
            include="number"
        )

            return numeric_df.mean().to_string()

        return "Sorry, I don't understand that question yet."
if __name__ == "__main__":
    app.run(debug=True)