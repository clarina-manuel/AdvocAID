from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from textblob import TextBlob
import torch

app = Flask(__name__)

#trained GPT-2 model
model_path = "./feedback_model"
model = GPT2LMHeadModel.from_pretrained(model_path)  
tokenizer = GPT2Tokenizer.from_pretrained(model_path)  
tokenizer.pad_token = tokenizer.eos_token

#generate feedback using prompt used to explain the task to the model
def generate_feedback(input_text, model, tokenizer):
    #prompt (same format as training)
    prompt = f"Provide 2 sentences of advice to present this issue more confidently (say: you should provide more evidence, expand in greater detail, etc): {input_text}"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    
    #generate the output from the model
    output = model.generate(
        inputs['input_ids'],  
        max_length=1000,       
        num_return_sequences=1,  
        no_repeat_ngram_size=2,  
        temperature=0.6,     
        top_p=0.9,           
        top_k=50,            
        do_sample=True,      
        pad_token_id=tokenizer.eos_token_id  
    )

    generated_feedback = tokenizer.decode(output[0], skip_special_tokens=True)
    feedback = generated_feedback[len(prompt):].strip() #exclude repeated prompt from generated text
    
    return feedback

# post processing
def personalize_feedback(input_text, generated_feedback):
    #tailored feedback for salary, compensation, or promotion-related topics
    if any(keyword in input_text.lower() for keyword in ["raise", "compensation", "promotion", "salary"]):
        return " You might also want to consider exploring how your achievements align with the company's growth and industry standards. If your salary is below the market average for similar roles, this could further support your case for an increase or promotion."

    #tailored feedback for workload, responsibilities, or burnout-related topics
    if any(keyword in input_text.lower() for keyword in ["workload", "responsibilities", "burnout"]):
        return " It's important to express how the additional workload is impacting your ability to maintain quality or meet expectations. You could also suggest breaking down tasks, prioritizing urgent ones, and perhaps delegating or sharing responsibilities with others to reduce the pressure."

    #tailored feedback for work-life balance, personal time, or stress-related topics
    if any(keyword in input_text.lower() for keyword in ["well-being", "work-life balance", "personal", "manage", "stress"]):
        return " You might want to suggest options such as flexible working hours, remote work, or adjusting your responsibilities to regain a healthier balance. Be specific about how this change could improve both your productivity and overall well-being."

    #tailored feedback for office conditions or environment-related topics
    if any(keyword in input_text.lower() for keyword in ["conditions", "environment", "workspace"]):
        return " It could be helpful to mention specific ways in which improving the office environment—like adjusting lighting, noise levels, or ergonomics—could support both your health and productivity. Maybe consider proposing a survey or seeking feedback from other colleagues."

    #tailored feedback for timelines, deadlines, or schedule-related topics
    if any(keyword in input_text.lower() for keyword in ["timelines", "schedule", "deadline"]):
        return " If timelines are tight, you could suggest a more realistic schedule or request additional resources to meet the deadlines. It might help to highlight any potential risks to project quality or deadlines if adjustments aren't made soon."

    #default response
    return "Looks good! All the best as you tackle this conversation!"


#sentiment analysis function to assess confidence and self-advocacy
def sentiment_analysis(user_input):
    blob = TextBlob(user_input)
    sentiment_score = blob.sentiment.polarity
    
    if sentiment_score > 0.5:
        confidence = "High"
    elif sentiment_score > 0:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    #self-advocacy logic
    if "I deserve" in user_input or "I believe" in user_input or "need" in user_input or "concerned" in user_input:
        self_advocacy = "High"
    elif "I think" in user_input or "I'm pretty sure" in user_input:
        self_advocacy = "Medium"
    else:
        self_advocacy = "Low"
    
    return confidence, self_advocacy

#html pages
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/scenarios')
def scenarios():
    return render_template('scenarios.html')

@app.route('/experts')
def experts():
    return render_template('experts.html')

#handle user input and provide feedback
@app.route('/generate_feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    user_input = data['response']
    
    #generate feedback
    feedback = generate_feedback(user_input, model, tokenizer)
    processed_feedback = personalize_feedback(user_input, feedback)
    
    #sentiment analysis
    confidence, self_advocacy = sentiment_analysis(user_input)
    
    return jsonify({
        'feedback': feedback,
        'additional_feedback': processed_feedback,
        'confidence': confidence,
        'self_advocacy': self_advocacy
    })

if __name__ == "__main__":
    app.run(debug=True)