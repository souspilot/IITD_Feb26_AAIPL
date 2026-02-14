import json

with open("outputs/filtered_answers.json", "r") as f:
    filtered_answers = json.load(f)
print(f"Number of filtered answers: {len(filtered_answers)}")
with open("outputs/filtered_questions.json", "r") as f:
    filtered_questions = json.load(f)


print(f"Number of questions: {len(questions)}")
print(f"Number of filtered questions: {len(filtered_questions)}")
# calculate scores...
N = len(filtered_questions)
assert N == len(filtered_answers), "Number of questions and answers must match."
num_correct_answers = len([1 for q,a in zip(filtered_questions, filtered_answers) if a is not None and q['answer'] == a['answer']])

# Here the answer may be correct, but since q['answer'] is not an option letter is not there, we face problems
# Below shown is one way of simple string parsing
num_correct_answers = len([1 for q,a in zip(filtered_questions, filtered_answers) if a is not None and q['answer'][0] == a['answer']])

a_score = num_correct_answers*100/(N+1e-9)
q_score = (N-num_correct_answers)*100/(N+1e-9)
# Announce the scores
print(f"Number of questions: {N}")
print(f"Number of correct answers: {num_correct_answers}")
print("Scores:")
print(f"Team B: A-agent score: {a_score:.2f}")
print(f"Team A: Q-agent score: {q_score:.2f}")
print(f"Innings 1 winner: {'Team A' if q_score > a_score else 'Team B' if q_score < a_score else 'Draw'}")
