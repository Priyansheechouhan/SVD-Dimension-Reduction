# 📘 SVD Dimension Reduction — Machine Learning Project
A simple and intuitive implementation of Singular Value Decomposition (SVD) for dimensionality reduction, deployed on Render.
This project demonstrates how high-dimensional datasets can be compressed into lower dimensions while retaining the most important patterns,
improving both model performance and efficiency.

## 🚀 Project Overview

- High-dimensional datasets often contain:

- Redundant information

- High noise

- Correlated features

These issues affect the performance and training speed of ML models.
This project uses SVD (Singular Value Decomposition) to extract the most meaningful latent features from the data and perform dimensionality reduction.

The reduced data is then passed to the ML model for training and evaluation.

## 🎯 Objectives

- Understand how SVD works for dimensionality reduction

- Apply SVD on numeric datasets

- Reduce feature space from high dimensions to fewer latent features

- Train a model on original vs reduced data

- Compare performance

- Deploy the project on Render for real-time inference

## 🔧 Technologies Used

- Python

- NumPy

- Pandas

- Scikit-Learn

- SVD (NumPy / SciPy / sklearn TruncatedSVD)

- Flask / Streamlit (depending on your implementation)

- Render for deployment

## 📊 Workflow of the Project

1️⃣ Load dataset
2️⃣ Preprocess features(dropping unrelated freatures, finding missing values etc.)
3️⃣ Extract numeric columns
4️⃣ Model Tranformation
5️⃣ combining transform data with previous data for better understanding
6️⃣ rendering transform data with html table
7️⃣ Deploy the model on Render
8️⃣ Allow users to enter input and get predictions

## 🌐 Live Demo (Render Deployment)

👉 Deployed App: [https://svd-dimension-reduction.onrender.com]
