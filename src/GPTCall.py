"""
Room Type Classification Module

This module provides functionality to classify room types using OpenAI API.
It serves as an interface that can be integrated into larger systems.
"""

import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Optional

# Initialize OpenAI client
load_dotenv()
client = OpenAI()


def classify_room_types(
    input_data: pd.DataFrame,
    reference_classes: List[str],
    model: str = "gpt-5-nano",
    batch_size: int = 20,
    reasoning_effort: str = "low",
    verbosity: str = "low"
) -> pd.DataFrame:
    """
    Classify room types by mapping architect room names to reference classes.
    
    Args:
        input_data (pd.DataFrame): DataFrame with columns:
            - 'Architect room names': Room type labels from architects
            - 'Area': Room area in square meters
            - 'Volume': Room volume (not used in classification, but preserved in output)
        reference_classes (List[str]): List of company's room type labels to map to
        model (str): OpenAI model to use (default: "gpt-5-nano")
        batch_size (int): Number of items to process per API call (default: 20)
        reasoning_effort (str): Reasoning effort level (default: "low")
        verbosity (str): Response verbosity level (default: "low")
    
    Returns:
        pd.DataFrame: DataFrame with columns:
            - 'Architect room names': Original architect room names
            - 'Area': Room area
            - 'Volume': Room volume
            - 'BKW_name': Predicted classification (mapped to reference classes)
    
    Raises:
        ValueError: If required columns are missing from input_data
        Exception: If API call fails
    """
    # Validate input
    required_columns = ['Architect room names', 'Area', 'Volume']
    missing_columns = [col for col in required_columns if col not in input_data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Get unique input-area combinations to avoid redundant processing
    df_unique = input_data.groupby('Architect room names', as_index=False).first()
    input_records = df_unique[['Architect room names', 'Area']].to_dict('records')
    
    print(f"  Total unique inputs: {len(input_records)}")
    
    # Process in batches
    all_mappings = []
    
    for batch_idx in range(0, len(input_records), batch_size):
        batch = input_records[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        total_batches = (len(input_records) + batch_size - 1) // batch_size
        
        print(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} items)...")
        
        # Build the prompt for this batch
        classes_str = '\n      '.join([f'"{c}",' for c in reference_classes])
        input_str = '\n'.join([f'{item["Architect room names"]} (Area: {item["Area"]} m²)' for item in batch])
        
        prompt = f"""### Role
You are a planning expert for a construction planning (civil engineering) company.

### Task
For each label in the input list, map the input room type labels (given by architects) to the best matching room type label from the company's own set.

### Input
1. Input Data: A list of input room type labels (architect) with their area in square meters
2. Classes: Set of construction company's room type labels

### Approach 
- Go through each item in the input list and assign it the best matching room type from classes by using your world knowledge and anticipating abbreviations used in the industry
- Use the area information as an additional context clue for classification
- Each input needs to be assigned exactly one class, classes can be assigned multiple times

### Output
Return a JSON array where each element is an object with "input" (the original room type) and "prediction" (the mapped class).

## Classes 
"classes": [
      {classes_str}
    ] 

## Input List
{input_str}
"""
        
        # Call OpenAI API with structured output
        response = client.responses.create(
            model=model,
            input=prompt,
            reasoning={"effort": reasoning_effort},
            text={
                "verbosity": verbosity,
                "format": {
                    "type": "json_schema",
                    "name": "room_type_mapping",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "mappings": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "input": {
                                            "type": "string",
                                            "description": "The original room type label from architect"
                                        },
                                        "prediction": {
                                            "type": "string",
                                            "description": "The predicted class from company labels"
                                        }
                                    },
                                    "required": ["input", "prediction"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["mappings"],
                        "additionalProperties": False
                    }
                }
            },
        )
        
        # Get the response content and parse JSON
        result_json = json.loads(response.output_text)
        batch_mappings = result_json['mappings']
        
        # Clean up the input field - remove area information if present
        for mapping in batch_mappings:
            if 'input' in mapping and ' (Area:' in mapping['input']:
                mapping['input'] = mapping['input'].split(' (Area:')[0]
        
        all_mappings.extend(batch_mappings)
        
        print(f"    ✓ Batch {batch_num} completed ({len(batch_mappings)} mappings)")
    
    # Convert mappings to DataFrame
    mappings_df = pd.DataFrame(all_mappings)
    mappings_df.rename(columns={'input': 'Architect room names', 'prediction': 'BKW_name'}, inplace=True)
    
    # Merge predictions back with original data
    result_df = input_data.merge(
        mappings_df[['Architect room names', 'BKW_name']],
        on='Architect room names',
        how='left'
    )
    
    # Reorder columns to match expected output format
    result_df = result_df[['Architect room names', 'Area', 'Volume', 'BKW_name']]
    
    print(f"  Total mappings collected: {len(mappings_df)}")
    
    return result_df


def classify_from_csv(
    input_csv_path: str,
    reference_csv_path: str,
    output_csv_path: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Classify room types from a CSV file using reference classes from another CSV.
    
    Args:
        input_csv_path (str): Path to input CSV file with columns:
            - 'Architect room names': Room type labels from architects
            - 'Area': Room area in square meters
            - 'Volume': Room volume
        reference_csv_path (str): Path to reference CSV file with column:
            - 'RoomType': Standard room type labels
        output_csv_path (Optional[str]): Path to save output CSV. If None, doesn't save.
        **kwargs: Additional arguments to pass to classify_room_types()
            - model (str): OpenAI model to use (default: "gpt-4o-mini")
            - batch_size (int): Number of items per API call (default: 20)
            - reasoning_effort (str): Reasoning effort for o1/o3 models (default: "low")
            - verbosity (str): Response verbosity (default: "low")
    
    Returns:
        pd.DataFrame: Classified data with columns:
            - 'Architect room names': Original architect room names
            - 'Area': Room area
            - 'Volume': Room volume
            - 'BKW_name': Predicted classification (mapped to reference classes)
    """
    # Load input data
    input_data = pd.read_csv(input_csv_path)
    print(f"✓ Loaded {len(input_data)} room records from {input_csv_path}")
    
    # Load reference classes from CSV
    reference_df = pd.read_csv(reference_csv_path)
    reference_classes = reference_df['RoomType'].unique().tolist()
    print(f"✓ Loaded {len(reference_classes)} reference room types from {reference_csv_path}")
    
    # Classify
    result_df = classify_room_types(input_data, reference_classes, **kwargs)
    
    # Save if output path provided
    if output_csv_path:
        result_df.to_csv(output_csv_path, index=False, encoding='utf-8')
        print(f"✓ Saved results to {output_csv_path}")
    
    return result_df


