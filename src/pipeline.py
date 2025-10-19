"""
HVAC Cost Estimation Orchestration Script

Pipeline:
1. Input: Architect room names with area and volume
2. Classify rooms using GPT to reference list
3. Calculate power requirements based on reference data
4. Get cost estimates using costCalculations
"""

import __main__
import pandas as pd
from src.GPTCall import classify_from_csv
from src.costCalculations import OllamaClassifier


def load_reference_classes(reference_csv: str) -> list:
    """Load reference room types from CSV"""
    df = pd.read_csv(reference_csv)
    # Extract unique room types from RoomType column
    return df['RoomType'].unique().tolist()


def get_room_mappings(
    input_csv: str,
    reference_csv: str,
    output_csv: str = "data/classified_rooms.csv",
) -> pd.DataFrame:
    """
    Step 1 & 2: Classify architect room names to reference classes
    
    Args:
        input_csv: Path to input CSV (columns: 'Architect room names', 'Area', 'Volume')
        reference_csv: Path to reference CSV
        output_csv: Path to save classified results
        model: GPT model to use
        batch_size: Batch size for processing
    
    Returns:
        DataFrame with columns: ['Architect room names', 'Area', 'Volume', 'BKW_name']
    """
    print("=" * 60)
    print("STEP 1 & 2: Room Classification")
    print("=" * 60)
    
    # Load input data
    input_data = pd.read_csv(input_csv)
    print(f"✓ Loaded {len(input_data)} room records")
    
    # Classify using GPT
    classified_df = classify_from_csv(
        input_csv_path=input_csv,
        reference_csv_path=reference_csv,
        output_csv_path=output_csv,
    )
    
    print(f"✓ Classified rooms saved to {output_csv}")
    print(f"\nSample mappings:")
    print(classified_df.head())
    
    return classified_df


def calculate_power_requirements(
    classified_df: pd.DataFrame,
    reference_csv: str
) -> pd.DataFrame:
    """
    Step 3: Calculate power requirements based on reference data
    
    Args:
        classified_df: DataFrame with classified rooms
        reference_csv: CSV with power requirements per m²
            Columns: RoomType, Heating_W_per_m2, Cooling_W_per_m2, Ventilation_m3_h_per_person, etc.
    
    Returns:
        DataFrame with added columns for KLT, HLT, RLT requirements
    """
    print("\n" + "=" * 60)
    print("STEP 3: Power Requirements Calculation")
    print("=" * 60)
    
    # Load reference data
    reference_df = pd.read_csv(reference_csv)
    print(f"✓ Loaded reference data for {len(reference_df)} room types")
    
    # Merge with classified data
    result_df = classified_df.merge(
        reference_df,
        left_on='BKW_name',
        right_on='RoomType',
        how='left'
    )
    
    # Calculate power requirements
    # Convert W to KW (divide by 1000)
    result_df['HLT_required_KW'] = (result_df['Area'] * result_df['Heating_W_per_m2']) / 1000
    result_df['KLT_required_KW'] = (result_df['Area'] * result_df['Cooling_W_per_m2']) / 1000
    
    # For ventilation, we need to estimate number of people
    # Assuming 10 m² per person as a standard (you can adjust this)
    result_df['estimated_persons'] = result_df['Area'] / 10
    result_df['RLT_required_m3h'] = result_df['estimated_persons'] * result_df['Ventilation_m3_h_per_person']
    
    print(f"✓ Calculated power requirements")
    print(f"\nSample calculations:")
    print(result_df[['BKW_name', 'Area', 'KLT_required_KW', 'HLT_required_KW', 'RLT_required_m3h']].head())
    
    return result_df


def estimate_costs(
    power_df: pd.DataFrame,
    model_name: str = "granite3.3:2b"
) -> dict:
    """
    Step 4: Get cost estimates for equipment using Ollama
    
    Args:
        power_df: DataFrame with power requirements
        model_name: Ollama model name for cost calculation
    
    Returns:
        Dictionary with cost breakdowns for KLT, HLT, RLT
    """
    print("\n" + "=" * 60)
    print("STEP 4: Cost Estimation")
    print("=" * 60)
    
    # Calculate total requirements
    total_klt = power_df['KLT_required_KW'].sum()
    total_hlt = power_df['HLT_required_KW'].sum()
    total_rlt = power_df['RLT_required_m3h'].sum()
    
    print(f"\nTotal Requirements:")
    print(f"  KLT (Cooling): {total_klt:.2f} KW")
    print(f"  HLT (Heating): {total_hlt:.2f} KW")
    print(f"  RLT (Ventilation): {total_rlt:.2f} m³/h")
    
    # Initialize cost calculator
    calculator = OllamaClassifier(model_name=model_name)
    
    # Get cost estimates for each type
    results = {}
    
    print("\n--- KLT (Cooling) Cost Estimate ---")
    results['KLT'] = calculator.generate(type="KLT", input=int(total_klt))
    print(results['KLT'])
    
    print("\n--- HLT (Heating) Cost Estimate ---")
    results['HLT'] = calculator.generate(type="HLT", input=int(total_hlt))
    print(results['HLT'])
    
    print("\n--- RLT (Ventilation) Cost Estimate ---")
    results['RLT'] = calculator.generate(type="RLT", input=int(total_rlt))
    print(results['RLT'])
    
    return results


def run_pipeline(
    input_csv: str,
    reference_csv: str,
    output_dir: str = "output",
    ollama_model: str = "granite3.3:2b",
):
    """
    Main orchestration function
    
    Args:
        input_csv: Path to input CSV with architect room names
        reference_csv: Path to reference CSV
        output_dir: Directory to save output files
        gpt_model: GPT model for classification
        ollama_model: Ollama model for cost estimation
        batch_size: Batch size for processing
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("HVAC COST ESTIMATION PIPELINE")
    print("=" * 60)
    
    # Step 1 & 2: Get room mappings
    reference_classes = load_reference_classes(reference_csv)
    print(f"✓ Loaded {len(reference_classes)} reference room types")
    
    classified_df = get_room_mappings(
        input_csv=input_csv,
        reference_csv=reference_csv,
        output_csv=f"{output_dir}/classified_rooms.csv",
    )
    
    # Step 3: Calculate power requirements
    power_df = calculate_power_requirements(
        classified_df=classified_df,
        reference_csv=reference_csv
    )
    power_df.to_csv(f"{output_dir}/power_requirements.csv", index=False)
    
    # Step 4: Get cost estimates
    cost_results = estimate_costs(power_df, model_name=ollama_model)
    
    # Save cost results
    with open(f"{output_dir}/cost_estimates.txt", 'w') as f:
        f.write("HVAC COST ESTIMATES\n")
        f.write("=" * 60 + "\n\n")
        f.write("KLT (Cooling Equipment)\n")
        f.write("-" * 60 + "\n")
        f.write(cost_results['KLT'] + "\n\n")
        f.write("HLT (Heating Equipment)\n")
        f.write("-" * 60 + "\n")
        f.write(cost_results['HLT'] + "\n\n")
        f.write("RLT (Ventilation Equipment)\n")
        f.write("-" * 60 + "\n")
        f.write(cost_results['RLT'] + "\n")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"✓ Results saved to {output_dir}/")
    
    return classified_df, power_df, cost_results

# if __name__ == "__main__":
#     # Example usage
#     run_pipeline(
#         input_csv="data/input.csv",
#         reference_csv="data/reference.csv",
#         output_dir="output",
#         ollama_model="granite3.3:2b",
#     )