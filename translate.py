import torch
import time
from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-bn-en"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("🚀 Starting Translation Pipeline")
print(f"🖥️ Device in use: {device}")
print("=" * 60)

start_time = time.time()

# Load model
print("📥 Loading tokenizer...")
tokenizer = MarianTokenizer.from_pretrained(model_name)

print("📥 Loading model...")
model = MarianMTModel.from_pretrained(model_name).to(device)

print("✅ Model loaded successfully!")

def translate_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    translated_lines = []

    print(f"\n📄 Total sentences: {len(lines)}")
    print("-" * 60)

    for i, line in enumerate(lines):
        line = line.strip()

        if not line:
            translated_lines.append("")
            continue

        print(f"🔄 Processing sentence {i+1}/{len(lines)}")
        print(f"   Input: {line}")

        inputs = tokenizer(line, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            translated = model.generate(**inputs)

        tgt_text = tokenizer.decode(translated[0], skip_special_tokens=True)

        print(f"   Output: {tgt_text}\n")

        translated_lines.append(tgt_text)

    with open(output_path, "w", encoding="utf-8") as f:
        for line in translated_lines:
            f.write(line + "\n")

    print("-" * 60)
    print(f"💾 Output saved to: {output_path}")

if __name__ == "__main__":
    translate_file("input.txt", "output.txt")

    end_time = time.time()
    print("=" * 60)
    print(f"✅ Finished in {end_time - start_time:.2f} seconds")
    print("=" * 60)
