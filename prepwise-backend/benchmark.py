import os
import csv
import time
import requests
import statistics
from dotenv import load_dotenv
from llm_client import get_client

load_dotenv()

FASTAPI_BASE = "http://localhost:8000"

# ── 20 benchmark prompts ──────────────────────────────────────
PROMPTS = [
    # RAG-style (require resume context)
    "Generate one interview question about building a Random Forest classifier.",
    "Generate one interview question about deploying ML models on Android devices.",
    "Generate one interview question about reducing false negatives in classification.",
    "Generate one interview question about feature engineering for behavioral data.",
    "Generate one interview question about Flask REST API design for ML backends.",
    # system design
    "Generate one ML system design interview question about real-time inference.",
    "Generate one interview question about choosing between RAG and fine-tuning.",
    "Generate one interview question about monitoring ML models in production.",
    "Generate one interview question about handling class imbalance in training data.",
    "Generate one interview question about latency vs accuracy trade-offs.",
    # deep learning
    "Generate one interview question about transformer attention mechanisms.",
    "Generate one interview question about tokenization strategies for LLMs.",
    "Generate one interview question about overfitting and regularization.",
    "Generate one interview question about transfer learning vs training from scratch.",
    "Generate one interview question about gradient descent optimization.",
    # behavioral
    "Generate one behavioral interview question about handling a failed ML experiment.",
    "Generate one behavioral interview question about explaining ML results to non-technical stakeholders.",
    "Generate one behavioral interview question about prioritizing ML tasks under deadlines.",
    "Generate one behavioral interview question about learning a new ML framework quickly.",
    "Generate one behavioral interview question about collaborating with data engineers.",
]

PROVIDERS = ["gemini", "llama", "mixtral"]
TEMPERATURE = 0.7


def run_benchmark(session_id: str):
    results = []

    for provider in PROVIDERS:
        print(f"\n{'='*50}")
        print(f"Provider: {provider.upper()}")
        print(f"{'='*50}")

        client = get_client(provider)

        for i, prompt in enumerate(PROMPTS):
            print(f"  [{i+1}/20] {prompt[:60]}...")

            try:
                start = time.perf_counter()
                result = client.chat(prompt, temperature=TEMPERATURE)
                latency = round(time.perf_counter() - start, 3)

                results.append({
                    "provider": provider,
                    "model": result["model"],
                    "prompt_id": i + 1,
                    "prompt": prompt,
                    "output": result["text"],
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "latency_s": latency,
                    "output_length": len(result["text"]),
                    "manual_score": ""  # fill in after
                })

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    "provider": provider,
                    "model": provider,
                    "prompt_id": i + 1,
                    "prompt": prompt,
                    "output": f"ERROR: {e}",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "latency_s": 0,
                    "output_length": 0,
                    "manual_score": ""
                })

            # rate limit protection
            if provider == "gemini":
                time.sleep(13)
            else:
                time.sleep(0.5)  # groq is generous

    # save raw results
    with open("benchmark_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ Saved to benchmark_results.csv")
    print(f"  Fill in 'manual_score' column (1-5) for each row, then run summarize.")


def summarize():
    by_provider: dict[str, list] = {}

    with open("benchmark_results.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row["provider"]
            if p not in by_provider:
                by_provider[p] = []
            by_provider[p].append(row)

    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")

    summary_rows = []

    for provider, rows in by_provider.items():
        latencies = [float(r["latency_s"]) for r in rows if float(r["latency_s"]) > 0]
        out_tokens = [int(r["output_tokens"]) for r in rows if int(r["output_tokens"]) > 0]
        scores = [int(r["manual_score"]) for r in rows if r["manual_score"].strip()]

        avg_latency = round(statistics.mean(latencies), 3) if latencies else 0
        avg_tokens = round(statistics.mean(out_tokens)) if out_tokens else 0
        avg_score = round(statistics.mean(scores), 2) if scores else "not scored"

        model_name = rows[0]["model"]

        print(f"\nProvider : {provider} ({model_name})")
        print(f"  avg quality score  : {avg_score} / 5")
        print(f"  avg latency        : {avg_latency}s")
        print(f"  avg output tokens  : {avg_tokens}")

        summary_rows.append({
            "provider": provider,
            "model": model_name,
            "avg_score": avg_score,
            "avg_latency_s": avg_latency,
            "avg_output_tokens": avg_tokens,
        })

    # save summary
    with open("benchmark_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n✓ Summary saved to benchmark_summary.csv")


if __name__ == "__main__":
    mode = input("Run benchmark or summarize? (b/s): ").strip().lower()
    if mode == "b":
        session_id = input("Enter session_id: ").strip()
        run_benchmark(session_id)
    else:
        summarize()