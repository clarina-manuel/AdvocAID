from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
import json

#pre-trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

tokenizer.pad_token = tokenizer.eos_token

#salary, workloads, work-life balance, projects + responsibilities
data = {
    "input": [
        "Provide feedback on this statement to help the user sound more confident and assertive. I believe my contributions to the team have been outstanding, and I’d like to discuss a salary increase.",
        "Provide feedback on this statement to help the user sound more confident and assertive. I’ve consistently exceeded my targets this year, and I think it’s time for a promotion.",
        "Provide feedback on this statement to help the user sound more confident and assertive. I’ve been with the company for two years now and feel I’ve earned a raise based on my performance.",
        "Provide feedback on this statement to help the user sound more confident and assertive. I’ve successfully led multiple high-profile projects, and I think it’s time to talk about a raise.",
        "Provide feedback on this statement to help the user sound more confident and assertive. I believe my role has evolved, and I’d like to revisit my compensation to reflect my added responsibilities.",
        "Provide feedback on this statement to help the user sound more confident and assertive. Given my recent achievements and leadership in the department, I’d like to discuss the next steps for my career growth.",
        "Provide feedback on this statement to help the user sound more confident and assertive. I’ve been with the company for a while now, and I feel my compensation doesn’t reflect the value I bring to the team.",
        "Provide feedback on this statement to help the user sound more confident and assertive. I believe the work I’ve done on recent projects should be rewarded with a salary increase. Let’s discuss.",
        "Provide feedback on this statement to help the user sound more confident and assertive. I’ve taken on a significant amount of additional responsibilities recently, and I’d like to discuss a raise.",
        "Provide feedback on this statement to help the user sound more confident and assertive. I believe the impact I’ve made in my current role qualifies me for a promotion. Let’s discuss the possibilities.",
        "Provide feedback on this statement to help the user sound more confident and assertive. I feel that my workload is too much for one person, and I would like to discuss redistributing some tasks.",
        "Provide feedback on this statement to help the user sound more confident and assertive. I’m currently managing several projects, and I feel I need additional support to meet deadlines effectively.",
        "Provide feedback on this statement to help the user sound more confident and assertive. I’ve been assigned additional tasks outside my role, and I would like to discuss adjusting my job description accordingly.",
        "Provide feedback on this statement to help the user sound more confident and assertive. My workload has increased significantly, and I’m finding it hard to balance everything. Let’s discuss a solution.",
        "Provide feedback on this statement to help the user sound more confident and assertive. I’m being asked to take on tasks that are outside of my expertise. I’d like to discuss how we can address this.",
        "Provide feedback on this statement to help the user sound more confident and assertive. I’m handling multiple responsibilities, and I’m concerned about burnout. Can we explore ways to better balance my workload?",
        "Provide feedback on this statement to help the user sound more confident and assertive. I feel that the scope of my role has expanded, and I’d like to discuss realigning my responsibilities to reflect that.",
        "Provide feedback on this statement to help the user sound more confident and assertive. I’m finding it difficult to manage both the quantity and quality of my workload. Can we discuss potential solutions?",
        "Provide feedback on this statement to help the user sound more confident and assertive. I’ve been asked to take on more responsibilities, but I think I need more time or resources to manage them effectively.",
        "Provide feedback on this statement to help the user sound more confident and assertive. I’m handling a larger workload than I anticipated, and I think it’s time to reassess my responsibilities.",
        "Provide feedback on this statement to help the user sound more confident and assertive. I feel like my work-life balance has been affected by the current workload, and I’d like to discuss options to improve it.",
        "Provide feedback on this statement to help the user sound more confident and assertive. I’ve been struggling to maintain a healthy work-life balance, and I’d like to explore some potential solutions.",
        "Provide feedback on this statement to help the user sound more confident and assertive. The demands of my role are starting to affect my personal time. Can we discuss adjusting my workload to address this?",
        "Provide feedback on this statement to help the user sound more confident and assertive. I’m finding it hard to manage both my personal life and my work responsibilities. Let’s discuss how we can improve the balance.",
        "Provide feedback on this statement to help the user sound more confident and assertive. I’ve been working overtime regularly, and I feel it’s affecting my work-life balance. Can we explore ways to address this?",
        "Provide feedback on this statement to help the user sound more confident and assertive. I’ve been unable to take time off lately, and I’m concerned about how this is impacting my well-being. Can we discuss my time off?",
        "Provide feedback on this statement to help the user sound more confident and assertive. I want to maintain a better work-life balance and would like to discuss how we can make this more feasible.",
        "Provide feedback on this statement to help the user sound more confident and assertive. I’m feeling overwhelmed with work and need to find a way to create more space for personal time. Let’s discuss solutions.",
        "Provide feedback on this statement to help the user sound more confident and assertive. The current workload is taking up much of my personal time. Can we talk about adjusting expectations to create a better balance?",
        "Provide feedback on this statement to help the user sound more confident and assertive. I feel that my current work schedule is affecting my ability to focus on personal matters. Can we reassess my workload?",
        "Provide feedback on this statement to help the user sound more confident and assertive. I’ve noticed the office environment is becoming less conducive to productivity. Can we discuss potential improvements?",
        "Provide feedback on this statement to help the user sound more confident and assertive. I believe the current workspace setup could be improved to better support team collaboration. Let’s explore possible solutions.",
        "Provide feedback on this statement to help the user sound more confident and assertive. The current working conditions are affecting my ability to focus. Can we discuss how to improve the work environment?",
        "Provide feedback on this statement to help the user sound more confident and assertive. I feel the office space is too noisy for focused work. Can we explore ways to reduce distractions?",
        "Provide feedback on this statement to help the user sound more confident and assertive. I believe we could improve the ergonomics of our workstations to better support employee health and productivity.",
        "Provide feedback on this statement to help the user sound more confident and assertive. The lighting in the office is not ideal, and I feel it’s impacting my comfort and work efficiency. Can we discuss a change?",
        "Provide feedback on this statement to help the user sound more confident and assertive. I think the office temperature could be better regulated to ensure a more comfortable work environment.",
        "Provide feedback on this statement to help the user sound more confident and assertive. The current break room setup isn’t ideal for relaxing during breaks. Let’s explore options for improvement.",
        "Provide feedback on this statement to help the user sound more confident and assertive. I believe having more natural light in the office would improve the overall atmosphere and productivity. Can we discuss this?",
        "Provide feedback on this statement to help the user sound more confident and assertive. The noise level in the office is often disruptive. Can we discuss ways to make the workspace quieter?",
        "Provide feedback on this statement to help the user sound more confident and assertive. I believe the current project deadline is unrealistic given the scope of work. Can we discuss extending it?",
        "Provide feedback on this statement to help the user sound more confident and assertive. I’m concerned that the project timeline is too tight, and I think we need to reassess the deadlines.",
        "Provide feedback on this statement to help the user sound more confident and assertive. The team is facing some challenges with the current project timeline. Can we discuss adjusting the expectations?",
        "Provide feedback on this statement to help the user sound more confident and assertive. I’m struggling to keep up with the project timeline. Could we revisit the deadlines or explore additional support?",
        "Provide feedback on this statement to help the user sound more confident and assertive. The current project timeline isn’t feasible with the current resources. Can we discuss how to adjust the schedule?",
        "Provide feedback on this statement to help the user sound more confident and assertive. I think we need to adjust the timeline for the project based on the complexity of the work required.",
        "Provide feedback on this statement to help the user sound more confident and assertive. The scope of the project has changed, and I believe we need to extend the deadline to accommodate the new requirements.",
        "Provide feedback on this statement to help the user sound more confident and assertive. Given the complexity of the project, I don’t believe we’ll meet the current deadline. Can we discuss adjusting the timeline?",
        "Provide feedback on this statement to help the user sound more confident and assertive. I’m concerned that the current timeline doesn’t account for potential delays. Can we reassess the schedule to be more realistic?",
        "Provide feedback on this statement to help the user sound more confident and assertive. I feel that the team could benefit from additional resources to meet the project deadline. Let’s discuss how we can improve support."
    ],
    "output": [
        "You’ve done a good job highlighting your achievements. Be more specific about the value you’ve added. For example: 'I’ve exceeded my targets by X%, and I believe this warrants a raise.'",
        "Instead of just asking for a promotion, explain why you deserve it. For example: 'I’ve consistently exceeded expectations, and I’m ready to take on more responsibilities. Let’s discuss the next step in my career.'",
        "Mention the specific results of your work over the past two years. For instance: 'In the last two years, I’ve delivered X and Y, and I believe this warrants a discussion about salary and promotion.'",
        "Focus on the results of the projects you’ve led. For example: 'The projects I’ve led have resulted in X outcome, and I believe my efforts should be recognized with a raise.'",
        "Instead of just asking for a raise, mention how your role has evolved. For example: 'I’ve taken on X, Y, and Z responsibilities, and I believe my compensation should reflect these changes.'",
        "Great job showcasing your leadership. Be more specific about the growth you’re seeking. For instance: 'Given my recent leadership, I’d like to explore opportunities for growth, including a potential raise or promotion.'",
        "You’ve done well to point out the value you bring. Be more specific about the difference between your current salary and market rates for your position. Say, 'I feel my compensation doesn’t align with industry standards for my role.'",
        "Instead of just saying you deserve a raise, tie it to concrete results. For example: 'I’ve contributed X and Y, and I believe a raise is appropriate based on these outcomes.'",
        "Provide evidence of how your added responsibilities have impacted the team or company. For example: 'Since taking on additional tasks, I’ve improved team efficiency by X%, and I believe this warrants a raise.'",
        "Clearly state why you're ready for a promotion. For example: 'I’ve been consistently exceeding my performance metrics, and I feel it’s time for a promotion to reflect my contributions.'",
        "You’ve done well to express your concerns. Be more specific about how the workload is affecting your performance. For example: 'The additional workload has made it difficult to maintain the quality of my work. Let’s discuss how we can redistribute tasks.'",
        "Make sure to explain the impact of your current workload. For instance: 'I’m managing several projects, and I’m concerned that without additional support, I won’t be able to meet deadlines.'",
        "It’s great that you’re addressing the extra tasks. Be clear about what responsibilities need to be adjusted. For example: 'I’ve taken on tasks outside my role, and I think it’s time to discuss formally adjusting my job description to reflect these changes.'",
        "Instead of just stating the problem, suggest a potential solution. For example: 'I’m overwhelmed by my current workload. Could we discuss redistributing some tasks or providing additional support to manage everything effectively?'",
        "Clarify the challenges you’re facing due to the tasks outside your expertise. For example: 'I’ve been asked to take on tasks outside my role, and I think it would be helpful to realign these duties to ensure quality.'",
        "Great job bringing up burnout. Suggest how the workload could be balanced. For instance: 'I’m concerned about burnout due to my current workload. Can we discuss adjusting my responsibilities or bringing in additional resources to support me?'",
        "Provide examples of how your role has expanded. For example: 'Over the past few months, I’ve taken on X and Y responsibilities. I’d like to discuss how we can realign my job to reflect this shift.'",
        "Instead of just mentioning the problem, offer a solution. For example: 'My workload has increased significantly, and I’m finding it challenging to maintain the quality of my work. Can we discuss ways to better distribute tasks or hire additional help?'",
        "State the specific resources you need to manage the additional responsibilities. For example: 'I’ve been asked to take on several new tasks, but I think I need more time or resources to manage them effectively. Can we explore potential solutions?'",
        "Be clear about the need for reassessment. For instance: 'I’ve been handling a larger workload than anticipated, and I think it’s time to reassess my responsibilities to ensure they align with my capabilities.'",
        "Great job addressing the issue. Be more specific about the impact on your well-being. For example: 'The current workload has significantly affected my work-life balance, and I’d like to discuss possible solutions to improve it.'",
        "Instead of just stating the problem, suggest solutions. For example: 'I’ve been struggling with work-life balance. Could we explore options like flexible working hours or adjusting my workload to address this?'",
        "Be more specific about how your role is affecting your personal time. For example: 'The demands of my role are starting to eat into my personal time, and I’d like to discuss adjusting my workload to find a better balance.'",
        "Explain how you’re trying to manage both your professional and personal life. For instance: 'I’m finding it difficult to maintain a healthy balance between my work and personal life. Could we discuss how to alleviate this pressure?'",
        "Clarify how the overtime is affecting you. For example: 'I’ve been working overtime regularly, and I think it’s affecting my work-life balance. Could we discuss options to ensure I have time for personal activities?'",
        "It’s great that you’re addressing time off. Be specific about how it affects your well-being. For instance: 'I’ve been unable to take time off, and I’m concerned it’s impacting my well-being. Could we discuss how to better manage time off moving forward?'",
        "State your intention for maintaining balance. For example: 'I’m committed to my role but also want to maintain a better work-life balance. Could we discuss how to make this more achievable for me?'",
        "Instead of just mentioning the overwhelm, offer a solution. For example: 'I’m feeling overwhelmed, and I’d like to discuss potential adjustments to my workload to ensure I have time for personal activities.'",
        "Provide specifics about how the current workload is impacting your life. For instance: 'The current workload is taking up much of my personal time, and I think adjusting expectations could help me regain a better balance.'",
        "Frame your request around your need for focus. For example: 'I feel my work schedule is interfering with my ability to focus on personal matters. Could we reassess my workload to ensure a better balance between both areas?'",
        "Great job noticing the need for improvement. Be specific about how the current environment affects your work. For example: 'The current office setup is distracting and negatively impacts my productivity. Let’s discuss potential improvements.'",
        "Instead of just mentioning the issue, suggest how it could be better. For example: 'The workspace setup could be improved for team collaboration. Could we explore options for more open spaces or dedicated collaboration areas?'",
        "Be specific about the impact on your focus. For example: 'The current working conditions are affecting my concentration, and I’d like to discuss changes to improve the office environment.'",
        "You’ve identified the problem clearly. Offer a potential solution, such as: 'The office is too noisy for focused work. Could we explore options for soundproofing or designated quiet areas?'",
        "Great approach in considering ergonomics. Be specific about the improvements. For example: 'I think ergonomic improvements to our workstations would reduce strain and improve overall productivity. Can we look into more adjustable desks and better chairs?'",
        "Be clear about the specific impact of lighting. For instance: 'The current office lighting makes it difficult to concentrate and impacts my comfort. Could we explore options for more balanced lighting?'",
        "Offer specific temperature-related improvements. For example: 'The office temperature fluctuates throughout the day, which makes it hard to focus. Could we explore options for better climate control?'",
        "Instead of just stating the problem, offer potential fixes. For example: 'The current break room setup doesn’t provide enough space for employees to relax. Could we look into improving the layout or adding more comfortable seating?'",
        "You’ve identified a great potential improvement with natural light. Be specific about how it would improve productivity. For instance: 'Having more natural light in the office would improve mood and productivity. Could we explore options to increase natural light in the workspace?'",
        "Instead of just pointing out the issue, offer solutions. For example: 'The noise level in the office is often disruptive. Could we look into noise-reducing materials or designate quiet zones for focused work?'",
        "You’ve done well to point out the issue. Offer potential solutions, for instance: 'I believe the current deadline is unrealistic given the project scope. Could we explore extending the deadline by a week to ensure the quality of work?'",
        "Great job raising your concern. Be specific about why the timeline is tight, for example: 'The current timeline is too tight because we haven’t yet completed phase X. Can we reassess and possibly adjust the deadline?'",
        "It’s good that you’re addressing team challenges. Be more specific about the challenges faced. For example: 'The team is encountering delays in X area. Could we explore extending the deadline to ensure the project is completed with the quality we expect?'",
        "Instead of just mentioning the struggle, suggest a solution. For instance: 'I’m finding it difficult to keep up with the timeline. Can we discuss ways to reallocate resources or extend the deadline to ensure success?'",
        "Be clear about the resource constraints. For example: 'With the current resources, I don’t think we can meet the deadline. Can we discuss options for additional support or a timeline extension?'",
        "Make sure to specify the complexity of the work. For example: 'Given the complexity of this project, I don’t believe the current timeline is feasible. Can we reassess the deadline to ensure we meet our objectives?'",
        "Instead of just stating the change in scope, suggest a new timeline. For example: 'The project scope has changed significantly, and I think we need to extend the deadline by X weeks to accommodate the new requirements.'",
        "Clarify why the timeline won’t be met. For instance: 'Based on the current project complexity, I don’t think we’ll meet the deadline. Could we reassess and possibly extend the timeline to ensure quality?'",
        "You’ve done well raising concerns about delays. Be more specific about what adjustments need to be made. For instance: 'I’m concerned the current timeline doesn’t account for potential delays, especially in phase X. Can we extend the deadline to mitigate risks?'",
        "Good job considering resources. For example: 'The team could benefit from additional support to meet the deadline. Could we look into bringing in temporary help or adjusting the schedule accordingly?'",
    ]
}

dataset = Dataset.from_dict(data)

#JSONL format
jsonl_data = []

for inp, out in zip(data["input"], data["output"]):
    jsonl_data.append({"prompt": inp, "completion": out})

with open('train_data.jsonl', 'w') as f:
    for entry in jsonl_data:
        f.write(json.dumps(entry) + '\n')



def tokenize_function(examples):
    encodings = tokenizer(examples['input'], padding="longest", truncation=True, max_length=512)
    encodings['labels'] = encodings['input_ids']
    return encodings

tokenized_dataset = dataset.map(tokenize_function, batched=True)

#train-test split (80/20)
train_test = tokenized_dataset.train_test_split(test_size=0.2, seed=123)

train_dataset = train_test['train']
test_dataset = train_test['test']

#training arguments
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,               
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=250,  
    save_steps=250,  
)

#trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

#fine-tune the model
trainer.train()

#evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation Results: {eval_results}")

#save
model.save_pretrained('./feedback_model')
tokenizer.save_pretrained('./feedback_model')