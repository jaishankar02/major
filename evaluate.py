import sacrebleu

def compute_bleu(output_file, reference_file):
    with open(output_file, "r", encoding="utf-8") as f:
        output = [line.strip() for line in f]

    with open(reference_file, "r", encoding="utf-8") as f:
        reference = [line.strip() for line in f]

    bleu = sacrebleu.corpus_bleu(output, [reference])
    print("BLEU Score:", bleu.score)

if __name__ == "__main__":
    compute_bleu("output.txt", "reference.txt")
