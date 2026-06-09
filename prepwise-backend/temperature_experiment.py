import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
import csv

load_dotenv()
gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ── same question, 3 temperatures ────────────────────────────
QUESTIONS = [
    "Generate one technical interview question about Random Forest classifiers.",
    "Generate one interview question about deploying ML models on mobile devices.",
    "Generate one behavioral interview question about handling model failures in production.",
]

TEMPERATURES = [0.3, 0.7, 1.0]

def generate_at_temp(prompt: str, temperature: float) -> tuple[str, float]:
    start = time.perf_counter()

    response = gemini.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            top_p=0.9,          # nucleus sampling — only top 90% probability mass
            max_output_tokens=200
        )
    )

    latency = round(time.perf_counter() - start, 3)
    return response.text.strip(), latency


def run_experiment():
    results = []

    for question in QUESTIONS:
        print(f"\n{'='*60}")
        print(f"PROMPT: {question}")
        print(f"{'='*60}")

        for temp in TEMPERATURES:
            output, latency = generate_at_temp(question, temp)

            print(f"\n  temp={temp} | latency={latency}s")
            print(f"  OUTPUT: {output[:120]}...")

            results.append({
                "prompt": question,
                "temperature": temp,
                "output": output,
                "latency_s": latency,
                "output_length": len(output),
                "manual_score": ""   # you fill this in after
            })

            time.sleep(13)  # avoid rate limiting

    # save to CSV
    keys = results[0].keys()
    with open("temperature_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ Results saved to temperature_results.csv")
    print(f"  Open it and fill in the 'manual_score' column (1-5) for each output.")
    return results


def summarize(csv_path: str = "temperature_results.csv"):
    """Run after filling in manual_score column."""
    import statistics

    by_temp: dict[float, list] = {0.3: [], 0.7: [], 1.0: []}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            temp = float(row["temperature"])
            score = row["manual_score"].strip()
            if score:
                by_temp[temp].append({
                    "score": int(score),
                    "latency": float(row["latency_s"]),
                    "length": int(row["output_length"])
                })

    print(f"\n{'='*60}")
    print("TEMPERATURE EXPERIMENT SUMMARY")
    print(f"{'='*60}")

    for temp, entries in by_temp.items():
        if not entries:
            continue
        scores = [e["score"] for e in entries]
        latencies = [e["latency"] for e in entries]
        lengths = [e["length"] for e in entries]

        print(f"\ntemperature={temp}")
        print(f"  avg quality score : {round(statistics.mean(scores), 2)} / 5")
        print(f"  avg latency       : {round(statistics.mean(latencies), 3)}s")
        print(f"  avg output length : {round(statistics.mean(lengths))} chars")


if __name__ == "__main__":
    mode = input("Run experiment or summarize? (e/s): ").strip().lower()
    if mode == "e":
        run_experiment()
    else:
        summarize()