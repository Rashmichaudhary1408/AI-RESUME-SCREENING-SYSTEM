
import gradio as gr
import fitz
import joblib
import re
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "resume_classifier_model.pkl"

model = joblib.load(MODEL_PATH)

SKILLS = [
    "python", "java", "c++", "sql", "machine learning",
    "deep learning", "artificial intelligence", "data analysis",
    "excel", "power bi", "tableau", "aws", "azure",
    "html", "css", "javascript", "react", "node.js",
    "communication", "project management", "leadership",
    "accounting", "marketing", "sales", "recruitment",
    "autocad", "civil engineering", "healthcare"
]

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def find_skills(text):
    text = text.lower()
    found = []
    for skill in SKILLS:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, text):
            found.append(skill)
    return found

def analyze_resume(pdf_path, job_description):
    if not pdf_path:
        return "Please upload a PDF resume.", "—", "—", "—"
    if not job_description or not job_description.strip():
        return "Please enter a job description.", "—", "—", "—"

    try:
        with fitz.open(pdf_path) as doc:
            resume_text = " ".join(page.get_text() for page in doc)

        if not resume_text.strip():
            return "No readable text found in this PDF.", "—", "—", "—"

        cleaned_resume = clean_text(resume_text)
        category = model.predict([cleaned_resume])[0]

        vectorizer = model.named_steps["tfidf"]
        resume_vector = vectorizer.transform([cleaned_resume])
        job_vector = vectorizer.transform([clean_text(job_description)])
        similarity = cosine_similarity(resume_vector, job_vector)[0][0] * 100

        resume_skills = find_skills(resume_text)
        required_skills = find_skills(job_description)
        matched = [s for s in required_skills if s in resume_skills]
        missing = [s for s in required_skills if s not in resume_skills]

        return (
            "Analysis complete. Please review the original resume.",
            category,
            f"{similarity:.2f}%",
            f"MATCHED: {', '.join(matched) or 'None detected'}\n\n"
            f"NOT FOUND IN TEXT: {', '.join(missing) or 'None detected'}"
        )
    except Exception as e:
        return f"Could not analyze resume: {e}", "—", "—", "—"

CSS = """
.gradio-container { max-width: 1150px !important; margin: auto !important; }
#hero {
  background: linear-gradient(120deg, #10172d, #27205a 65%, #3b246b);
  border: 1px solid #454078; border-radius: 22px;
  padding: 30px; color: white; margin-bottom: 18px;
}
#hero h1 { font-size: 35px; margin: 8px 0; color: white; }
#hero p { color: #cbd5f5; font-size: 15px; }
#analyze {
  background: linear-gradient(90deg, #7557ee, #9b59ee);
  color: white; border: none; border-radius: 12px;
  font-weight: 700; min-height: 48px;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="indigo",
        neutral_hue="slate"
    ),
    css=CSS,
    title="TalentLens AI | Resume Intelligence"
) as demo:
    gr.HTML("""
    <div id="hero">
      <div style="font-size:12px;letter-spacing:3px;color:#c4b5fd">
        ✦ AI-POWERED RECRUITMENT WORKSPACE
      </div>
      <h1>TalentLens AI</h1>
      <p>Analyze resumes, compare job requirements, and identify
      relevant skills in one workspace.</p>
      <span>◈ 24 Resume Categories &nbsp; · &nbsp;
      ◉ PDF Analysis &nbsp; · &nbsp; ✧ Skill Matching</span>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("## 01 · Candidate Resume")
            resume_file = gr.File(
                label="Upload resume",
                file_types=[".pdf"],
                type="filepath"
            )
            gr.Markdown("Text-based PDF recommended.")
        with gr.Column(scale=7):
            gr.Markdown("## 02 · Job Requirements")
            job_input = gr.Textbox(
                label="Job description",
                lines=9,
                placeholder="Enter the job title, requirements, and skills..."
            )

    analyze_btn = gr.Button("✦  Analyze Resume", elem_id="analyze")
    gr.Markdown("## 03 · Analysis Dashboard")
    status_output = gr.Textbox(label="Analysis status", interactive=False)

    with gr.Row():
        category_output = gr.Textbox(label="Predicted category", interactive=False)
        similarity_output = gr.Textbox(
            label="Job-description similarity",
            interactive=False
        )

    skills_output = gr.Textbox(
        label="Skill matching results", lines=4, interactive=False
    )

    analyze_btn.click(
        fn=analyze_resume,
        inputs=[resume_file, job_input],
        outputs=[
            status_output, category_output,
            similarity_output, skills_output
        ]
    )

    gr.Markdown(
        "**Responsible use:** Similarity is not a probability of qualification. "
        "Use results to support human review, not automatic hiring or rejection."
    )

if __name__ == "__main__":
    demo.launch()
