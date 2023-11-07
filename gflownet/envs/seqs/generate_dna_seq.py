import random
import pandas as pd

# Constants
NUCLEOBASES = ['A', 'C', 'G', 'T']
#SEQUENCE_COUNT = 100000
MIN_LENGTH = 10
MAX_LENGTH = 64
N_SAMPLE = 1000000
# Prompt the user for a specific substring
specific_substring = 'AGGCT'
random.seed(99)
# Function to generate a random DNA sequence
def generate_random_dna_sequence(min_length, max_length):
    return ''.join(random.choices(NUCLEOBASES, k=random.randint(min_length, max_length)))

# Generate the dataset
dataset = {'sequence': [], 'contains_substring': []}
sequences_with_substring = []
sequences_without_substring = []

#for _ in range(SEQUENCE_COUNT):
while len(sequences_with_substring) < N_SAMPLE:
    sequence = generate_random_dna_sequence(MIN_LENGTH, MAX_LENGTH)
    contains_substring = 1 if specific_substring in sequence else 0
    dataset['sequence'].append(sequence)
    dataset['contains_substring'].append(contains_substring)
    
    # Separate the sequences into lists based on the presence of the substring
    if contains_substring:
        sequences_with_substring.append({'sequence': sequence, 'target': 1})
    else:
        sequences_without_substring.append({'sequence': sequence, 'target': 0})

#random sampling from sequences_without_substring
sequences_without_substring_sample = random.sample(sequences_without_substring, N_SAMPLE)

#Combine two lists and perform random shuffle
combined_list = sequences_with_substring + sequences_without_substring_sample
random.shuffle(combined_list)

print(len(sequences_with_substring), len(sequences_without_substring), len(sequences_without_substring_sample), len(combined_list))
# Convert to DataFrame
#df_with_substring = pd.DataFrame(sequences_with_substring)
#df_without_substring = pd.DataFrame(sequences_without_substring_sample)
df = pd.DataFrame(combined_list)
# Save to CSV files
#df_with_substring.to_csv('sequences_with_substring.csv', index=False)
#df_without_substring.to_csv('sequences_without_substring.csv', index=False)
df.to_csv('dna_sequences.csv', index=False)

