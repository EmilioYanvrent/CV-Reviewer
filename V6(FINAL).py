import torch
import os
import pdfplumber
from tkinter import Tk, filedialog
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from pymongo import MongoClient

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def connect_to_database():
    client = MongoClient("")#Fill in your own MongoClient
    return client["resume_analyzer"]

db = connect_to_database()
participants_collection = db["participants"]
jobs_collection = db["job_title"]

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text if text.strip() else None

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    offload_folder="offload"
)

generation_config = GenerationConfig.from_pretrained(model_name)
if generation_config.pad_token_id is None or generation_config.pad_token_id == generation_config.eos_token_id:
    tokenizer.pad_token = tokenizer.eos_token
    generation_config.pad_token_id = tokenizer.pad_token_id

model.generation_config = generation_config

# Get job title from database for the user to choose
def get_job_title_from_db():
    job_titles = list(jobs_collection.find({}, {"title": 1}))
    if not job_titles:
        print("No job titles found in the database.")
        return None
    print("Available Job Titles:")
    for i, job in enumerate(job_titles, start=1):
        print(f"{i}. {job['title']}")
    choice = input("Select the job title (number): ").strip()
    try:
        selected_job = job_titles[int(choice) - 1]
        return selected_job['title']
    except (ValueError, IndexError):
        print("Invalid choice. Please try again.")
        return None

# Match qualifications using LLM prompt
def match_qualifications(cv_text, job_qualifications):
    if not job_qualifications:
        return 0

    messages = [
        {"role": "system", "content": f"""
        You are an AI assistant specialized in evaluating job qualifications. 
        The following job qualifications are required for the position:

        {job_qualifications}

        Analyze the provided CV and determine if the candidate meets these qualifications. 
        Respond with a number score (0-10) indicating the match level, where:
        - 0 means no match at all.
        - 10 means a perfect match.
        """},
        {"role": "user", "content": f"CV content:\n{cv_text}"}
    ]

    input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    attention_mask = torch.ones(input_tensor.shape, dtype=torch.long).to(model.device)
    
    outputs = model.generate(
        input_tensor.to(model.device), 
        attention_mask=attention_mask,
        max_new_tokens=20,
        pad_token_id=model.generation_config.pad_token_id
    )
    
    result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)

    # Extract numerical score from the result
    try:
        score = int(result.strip())
    except ValueError:
        score = 0  # Default to 0 if the model output is invalid

    return score

# Match skills using word-based matching
def match_skills(cv_text, skills_needed):
    match_score = 0
    for word in skills_needed.split(','):
        if word.lower() in cv_text.lower():
            match_score += 1
    return match_score

# Determine suitability of the candidate
def determine_suitability(skills_match_score, qualifications_match_score, threshold=3):
    if skills_match_score > threshold and qualifications_match_score > threshold:
        return "The candidate is suitable for the position based on both skills and qualifications."
    elif skills_match_score > threshold:
        return "The candidate has the necessary skills but may need to improve on qualifications."
    elif qualifications_match_score > threshold:
        return "The candidate meets the qualifications but lacks some necessary skills."
    else:
        return "The candidate does not meet the required skills and qualifications for the position."

# Save or update the analysis to the database
def save_analysis_to_db(participant_name, job_title, skills_match_score, qualifications_match_score, suitability, result):
    # Check if the participant exists and matches the job title
    existing_participant = participants_collection.find_one({"name": participant_name, "job_title": job_title})

    if existing_participant:
        # If participant exists and job title matches, update the analysis
        update_result = participants_collection.update_one(
            {"name": participant_name, "job_title": job_title},  # Find participant by name and job title
            {"$set": {
                "skills_match_score": skills_match_score,
                "qualifications_match_score": qualifications_match_score,
                "suitability": suitability,
                "full_analysis": result
            }}
        )
        if update_result.matched_count > 0:
            print(f"Analysis for {participant_name} (Job: {job_title}) successfully updated.")
        else:
            print(f"Failed to update analysis for {participant_name}.")
    else:
        # If participant doesn't exist or job title doesn't match, insert a new record
        new_participant_data = {
            "name": participant_name,
            "job_title": job_title,
            "skills_match_score": skills_match_score,
            "qualifications_match_score": qualifications_match_score,
            "suitability": suitability,
            "full_analysis": result
        }
        insert_result = participants_collection.insert_one(new_participant_data)
        if insert_result.inserted_id:
            print(f"New analysis for {participant_name} (Job: {job_title}) successfully saved.")
        else:
            print(f"Failed to save new analysis for {participant_name}.")

# Analyze the CV with respect to the selected job title
def analyze_cv(cv_text, participant_name, job_title):
    print(f"Analyzing CV for: {participant_name} for position: {job_title}")
    
    job = jobs_collection.find_one({"title": job_title}, {"job_qualifications": 1, "skills": 1, "_id": 0})
    if not job:
        print("Job not found in database.")
        return None

    qualifications = job.get("job_qualifications", "")
    skills_needed = job.get("skills", "")
    
    # Compare skills
    skills_match_score = match_skills(cv_text, skills_needed)
    
    # Compare qualifications using LLM
    qualifications_match_score = match_qualifications(cv_text, qualifications)
    
    # Determine suitability based on the match scores
    suitability = determine_suitability(skills_match_score, qualifications_match_score)

    # Generate detailed analysis
    messages = [
        {"role": "system", "content": f"""
        The required job qualifications for this role are:

        {qualifications}

        The candidate's CV has been analyzed against these qualifications and skills.
        As an experienced recruiter, provide a structured analysis of the candidate’s profile, covering:

        1. **Summary**: A concise overview of the candidate’s background.
        2. **Strengths**: Key areas where the candidate excels.
        3. **Weaknesses**: Aspects where the candidate may need improvement.
        4. **Skills**: Relevant skills identified in the CV.

        Ensure your analysis is professional and to the point.
        """},
        {"role": "user", "content": f"Analyze this CV:\n{cv_text}"}
    ]

    input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    attention_mask = torch.ones(input_tensor.shape, dtype=torch.long).to(model.device)
    
    outputs = model.generate(
        input_tensor.to(model.device), 
        attention_mask=attention_mask,
        max_new_tokens=500,
        pad_token_id=model.generation_config.pad_token_id
    )
    
    result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)

    result += f"\n\nSkills Match Score: {skills_match_score}"
    result += f"\nQualifications Match Score: {qualifications_match_score}"
    result += f"\n\nSuitability: {suitability}"

    # Save the analysis to the database (either update or insert)
    save_analysis_to_db(participant_name, job_title, skills_match_score, qualifications_match_score, suitability, result)

    return result

# Analyze multiple CVs from a folder
def analyze_multiple_cvs():
    Tk().withdraw()
    folder_path = filedialog.askdirectory()
    if not folder_path:
        print("No folder selected. Exiting...")
        return
    files = [file for file in os.listdir(folder_path) if file.endswith('.pdf')]
    if not files:
        print("No PDF files found.")
        return
    job_title = get_job_title_from_db()
    if not job_title:
        return "ERROR"
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        participant_name = os.path.splitext(file_name)[0]
        cv_text = extract_text_from_pdf(file_path)
        if not cv_text:
            print(f"Skipping {file_name}: No text extracted.")
            continue
        analysis = analyze_cv(cv_text, participant_name, job_title)
        print(f"Analysis for {participant_name}: {analysis}")

# Main menu for interaction
def main_menu():
    while True:
        print("\nMenu:")
        print("1. Analyze multiple CVs")
        print("2. Exit")
        choice = input("Enter your selection: ").strip()
        if choice == "1":
            analyze_multiple_cvs()
        elif choice == "2":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Try again.")

main_menu()


# print(f'qualifications: {qualifications} \n skills_needed: {skills_needed}')
