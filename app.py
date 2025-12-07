# Importing necessary modules
from flask import Flask, render_template, request
import pandas as pd
from urllib.parse import quote
import joblib

# Creating a Flask application instance
app = Flask(__name__)

# Load the saved SVD model
model = joblib.load("svd_DimRed")

# Home page route
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling file upload & performing SVD transformation
@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':

        # ----------- 1. FILE VALIDATION -----------
        file = request.files.get("file")
        if not file or file.filename == "":
            return "No file uploaded", 400

        filename = file.filename.lower()
        print("Uploaded file name:", file.filename)

        # ----------- 2. FILE READING SAFELY -----------
        try:
            if filename.endswith(".csv"):
                data = pd.read_csv(file)
            elif filename.endswith(".xlsx"):
                data = pd.read_excel(file)
            else:
                return "Unsupported file type. Upload CSV or Excel only.", 400
        except Exception as e:
            print("Error reading file:", e)
            return f"Error reading file: {e}", 400

        print("Columns:", data.columns.tolist())
        print("Shape:", data.shape)

        if data.shape[0] == 0:
            return "Uploaded file is empty or unreadable.", 400

        # ----------- 3. PREPROCESSING (MATCH TRAINING PIPELINE) -----------
        # Drop UnivID column if present
        data = data.drop(['UnivID'], axis=1, errors='ignore')

        # Select only numeric columns used during training
        # Because SVD can ONLY transform numeric features
        num_cols = ['SAT', 'Top10', 'Accept', 'SFRatio', 'Expenses', 'GradRate']

        # Check if required numeric columns exist
        missing = [c for c in num_cols if c not in data.columns]
        print("Missing numeric columns:", missing)

        if missing:
            return f"Uploaded file missing required numeric columns: {missing}", 400

        data_num = data[num_cols]

        if data_num.shape[0] == 0:
            return "No valid numeric rows found for SVD transformation.", 400

        # ----------- 4. APPLY SVD MODEL SAFELY -----------
        try:
            svd_res = pd.DataFrame(
                model.transform(data_num),
                columns=['svd0', 'svd1', 'svd2', 'svd3', 'svd4', 'svd5']
            )
        except Exception as e:
            print("Model transformation error:", e)
            return f"Model transformation error: {e}", 400

        # ----------- 5. MERGE UNIV NAME + SVD RESULTS -----------
        if "Univ" in data.columns:
            final = pd.concat([data["Univ"], svd_res], axis=1)
        else:
            final = svd_res

        # Convert to HTML table
        try:
            html_table = final.to_html(classes='table table-striped')
        except Exception as e:
            print("HTML conversion error:", e)
            html_table = "<p>Error converting table to HTML.</p>"

        # ----------- 6. RETURN RESULT PAGE -----------
        return render_template(
            "data.html",
            Y=f"""
                <style>
                    .table {{
                        width: 70%;
                        margin: 20px auto;
                        border-collapse: collapse;
                    }}
                    .table thead {{
                        background-color: #39648f;
                        color: white;
                    }}
                    .table th, .table td {{
                        border: 1px solid #ddd;
                        padding: 8px;
                        text-align: center;
                    }}
                    .table td {{
                        background-color: #a8dfe3;
                    }}
                </style>
                {html_table}
            """
        )

# Run the application
if __name__ == '__main__':
    app.run(debug=True)
