from rouge import Rouge

# Example reference and generated summaries
references = [['the cat is on the mat'], ['there is a cat on the mat']]
hypothesis = 'the cat is on the mat'
# Initialize the Rouge object
rouge = Rouge();

# Calculate ROUGE scores
scores = rouge.get_scores(hypothesis, references)

print("scores->" + str(scores))
