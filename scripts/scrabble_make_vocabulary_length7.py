from pathlib import Path

vocabulary = []
with open(Path(__file__).parent / "../gflownet/utils/scrabble/vocabulary_en", "r") as f:
    vocabulary_all = f.readlines()
    for word in vocabulary_all:
        if len(word.rstrip()) <= 7:
            vocabulary.append(word)
print(f"Length of 7-letter vocabulary: {len(vocabulary)} / {len(vocabulary_all)}")
with open(
    Path(__file__).parent / "../gflownet/utils/scrabble/vocabulary_7letters_en", "w"
) as f:
    f.writelines(vocabulary)
