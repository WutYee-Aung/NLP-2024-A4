from flask import Flask, render_template, request
from utils import read_pdf, get_skills
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    resume_result = []

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file part")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No selected file")

        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            text = read_pdf(file_path)
            skill_dict = get_skills(text)

            # Extract skills for display
            resume_result = {
                'skills':skill_dict['skills'],
                'education':skill_dict['education'],
                'skill_dsai':skill_dict['skill_dsai'],
                'skill_bi':skill_dict['skill_bi'],
                'email':skill_dict['email'],
                'mobile':skill_dict['ph_no']
            }

            # Render the template with extracted skills
            return render_template('index.html', resume_result=resume_result)

    return render_template('index.html', resume_result=resume_result)

if __name__ == '__main__':
    app.run(debug=True)
