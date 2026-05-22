import os
import argparse
import pandas as pd
from src.generator import generate_synthetic_data, save_and_split
from src.evaluate import generate_static_tables, build_evaluation_tables_and_plots
from src.train import train_and_calibrate
from src.explain import get_tfidf_explanations

def main(days=365, seed=42, output_dir="outputs"):
    print("=" * 60)
    print("STARTING INDUSTRIAL O&M TRUSTWORTHY AI BENCHMARK PIPELINE")
    print("=" * 60)
    
    # Step 1: Generate synthetic data
    print("\n--- PHASE 1: Synthetic Data Generation ---")
    df, latent_df = generate_synthetic_data(days=days, seed=seed)
    
    # Step 2: Split data, run leakage audits, save Parquet
    print("\n--- PHASE 2: Leakage Auditing & Splitting ---")
    save_and_split(df, latent_df, output_dir=output_dir)
    
    # Step 3: Run static table definitions
    print("\n--- PHASE 3: Saving Schema and Methodology Tables ---")
    generate_static_tables(output_dir=output_dir)
    
    # Step 4: Run Experiments, Calibration, Routing, and Visualizations
    print("\n--- PHASE 4: Model Training, Experiment Execution & Visuals ---")
    build_evaluation_tables_and_plots(
        os.path.join(output_dir, "dataset.parquet"), 
        output_dir=output_dir
    )
    
    # Step 5: Post-run explanations (extract TF-IDF coefficients for transparent text baseline)
    print("\n--- PHASE 5: Post-run Text Explainability ---")
    # Load splits to train a quick local instance for TF-IDF extraction
    train_df = pd.read_parquet(os.path.join(output_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(output_dir, "val.parquet"))
    
    preprocessor, models, _, _ = train_and_calibrate(train_df, val_df)
    get_tfidf_explanations(models["lr_tfidf"], preprocessor.tfidf_vectorizer, output_dir=output_dir)
    
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETE")
    print(f"All artifacts, tables (1-12), and figures (1-11) saved to '{output_dir}/'")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()
    
    main(days=args.days, seed=args.seed, output_dir=args.output_dir)
