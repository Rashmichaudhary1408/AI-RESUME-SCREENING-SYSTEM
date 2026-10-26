# TalentLens AI — Resume Screening System

An AI-based resume screening prototype that classifies resumes
into job categories and compares resume text with a job description.

## Features
- PDF resume text extraction
- Resume category prediction
- TF-IDF text similarity
- Keyword-based skill matching
- Human-review-oriented results

## Machine Learning
- Dataset: 2,481 usable resumes across 24 categories
- Text features: TF-IDF
- Classifier: Logistic Regression
- Test accuracy: 64.79% on a held-out test set

## Tech Stack
Python, Scikit-learn, Gradio, PyMuPDF

## Run Locally
1. Install Python.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Place `resume_classifier_model.pkl` beside `app.py`.
4. Run:
   `python app.py`

## Limitations and Responsible Use
This is an educational prototype. Text similarity is not a
probability of qualification. Extracted text may omit skills,
and category predictions can be incorrect. Use results to
support human review, not automatic hiring or rejection.

## Project Status
Developed as an AI/ML internship project.
