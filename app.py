# Importing necessary modules
from flask import Flask, render_template, request  # Importing Flask module for creating web applications
import pandas as pd  # Importing Pandas library for data manipulation and analysis

# Creating a Flask application instance
app = Flask(__name__)

# Importing the saved SVD model using joblib
import joblib
model = joblib.load("svd_DimRed")


# Route for the home page of the Flask application
@app.route('/')
def home():
    # Rendering the 'index.html' template when the root URL is accessed
    return render_template('index.html')


# Route for handling file upload and performing prediction
@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        file = request.files.get("file")  # get the uploaded file

        if not file or file.filename == "":
            return "No file uploaded", 400

        filename = file.filename.lower()
        print("Uploaded file name:", filename)

        # ---- 1. Read the uploaded file safely (CSV / Excel) ----
        try:
            if filename.endswith(".csv"):
                data = pd.read_csv(file)
            elif filename.endswith(".xlsx") or filename.endswith(".xls"):
                data = pd.read_excel(file)
            else:
                return "Unsupported file format. Please upload CSV or Excel.", 400
        except Exception as e:
            print("Error reading file:", e)
            return f"Error reading file: {e}", 400

        print("Columns:", data.columns.tolist())
        print("Shape:", data.shape)

        # ---- 2. Ensure required columns are present ----
        # Columns in your dataset
        required_cols = ['Univ', 'SAT', 'Top10', 'Accept',
                         'SFRatio', 'Expenses', 'GradRate']
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            return f"Missing required columns in uploaded file: {missing}", 400

        # ---- 3. Drop UnivID if present, keep everything else ----
        if "UnivID" in data.columns:
            data1 = data.drop(["UnivID"], axis=1)
        else:
            data1 = data.copy()

        # ---- 4. Explicit numeric columns used when training the model ----
        num_cols = ['SAT', 'Top10', 'Accept', 'SFRatio', 'Expenses', 'GradRate']

        # Clean numeric columns: remove thousand separators like "21,864" -> "21864"
        for col in num_cols:
            # Convert to string, strip spaces, remove commas, then to numeric
            data1[col] = (
                data1[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            data1[col] = pd.to_numeric(data1[col], errors="coerce")

        print("Numeric dtypes after cleaning:")
        print(data1[num_cols].dtypes)

        # Drop rows where ALL numeric columns are NaN
        data_num = data1[num_cols].dropna(how="all")

        if data_num.shape[0] == 0:
            return "No valid numeric data after cleaning. Please check your file.", 400

        # ---- 5. Model transformation (SVD) ----
        try:
            svd_res = pd.DataFrame(
                model.transform(data_num),
                columns=['svd0', 'svd1', 'svd2', 'svd3', 'svd4', 'svd5']
            )
        except Exception as e:
            print("Model transformation error:", e)
            return f"Model transformation error: {e}", 400

        # Align the Univ column with data_num (in case some rows were dropped)
        univ_series = data.loc[data_num.index, 'Univ'].reset_index(drop=True)
        svd_res = svd_res.reset_index(drop=True)

        # ---- 6. Final result table ----
        final = pd.concat([univ_series.rename("Univ"), svd_res], axis=1)

        html_table = final.to_html(classes='table table-striped', index=False)

        # ---- 7. Render HTML with inline CSS ----
        return render_template(
            "data.html",
            Y=(
                "<style>\
                    .table {\
                        width: 50%;\
                        margin: 0 auto;\
                        border-collapse: collapse;\
                    }\
                    .table thead {\
                        background-color: #39648f;\
                    }\
                    .table th, .table td {\
                        border: 1px solid #ddd;\
                        padding: 8px;\
                        text-align: center;\
                    }\
                    .table td {\
                        background-color: #a8dfe3;\
                    }\
                    .table tbody th {\
                        background-color: #ab2c3f;\
                    }\
                </style>"
                + html_table
            )
        )


# Running the Flask application
if __name__ == '__main__':
    # Enabling debug mode for easier development
    app.run(debug=True)
