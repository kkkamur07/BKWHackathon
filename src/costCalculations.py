## Assuming Expected output to be KLT : KW ; HLT : KW and RLT : m^3/h

import ollama
import pandas as pd
import re

class OllamaClassifier():
    def __init__(self, model_name: str ):
        self.model_name = model_name

    def get_data(self, type: str) -> str:
        
        # Need to edit
        if type == "KLT" : 
            df = pd.read_csv("data/410.csv")
            return df.to_string(index=False)
        if type == "RLT" : 
            df = pd.read_csv("data/430.csv")
            return df.to_string(index=False)
        if type == "HLT" : 
            df = pd.read_csv("data/420.csv") 
            return df.to_string(index=False)

    def build_prompt(self, type : str, input : int) :
        if type == "KLT" : 
            context = self.get_data("KLT")
        if type == "RLT" :
            context = self.get_data("RLT")
        if type == "HLT" :
            context = self.get_data("HLT")
            
        prompt = f"""You are a cost calculation assistant specializing in HVAC equipment selection and pricing.
        
        **Equipment Type:** {type}
        - KLT: Cooling equipment (measured in KW)
        - HLT: Heating equipment (measured in KW)
        - RLT: Ventilation equipment (measured in m³/h)
        
        **Context - Available Equipment Data:**
        {context}
        
        **Task:**
        1. Analyze the {type} equipment data provided above
        2. Based on the input capacity of {input}, select the appropriate equipment from the list
        3. Calculate the total cost by aggregating all necessary components
        4. Consider the specific requirements for {type} systems
        
        **Input Capacity:** {input}
        
        **Required Output Format:**
        - List each selected equipment with its specifications and cost
        - Show the calculation breakdown
        - Provide total cost and capacity in the appropriate unit
        
        **Example Output:**
        Equipment 1: [Name] - [Specification] - Cost: $X
        Equipment 2: [Name] - [Specification] - Cost: $Y
        ---
        Total Cost: $Z
        {type}: [calculated capacity] {"KW" if type in ["KLT", "HLT"] else "m³/h"}
        
        Please analyze the data and provide the detailed cost breakdown with equipment selection.
        
        JUST OUTPUT THE FINAL ANSWER IN THE FORMAT SPECIFIED ABOVE. YOU SHOULDN'T HAVE ANY ADDITIONAL OUTPUT
        
        """
        return prompt
    
    def clean_output(self, output: str) -> str:
        """
        Clean the output by removing unwanted introductory and explanatory text.
        
        Args:
            output: Raw output from Ollama
        
        Returns:
            Cleaned output with only the cost breakdown
        """
        # Patterns to remove (case-insensitive)
        unwanted_patterns = [
            r"Based on.*?(?=Equipment|\n\n)",  # "Based on an input capacity..."
            r"Capacity:.*?\n",  # "Capacity: X KW" or "Capacity: X m³/h"
            r"we need to consider.*?(?=Equipment|\n\n)",  # explanatory sentences
            r"Here is.*?(?=Equipment|\n\n)",  # "Here is the breakdown..."
            r"The total cost.*?(?=Equipment|\n\n)",  # explanatory cost sentences before breakdown
            r"For.*?we will.*?(?=Equipment|\n\n)",  # "For X capacity, we will..."
            r"I'll.*?(?=Equipment|\n\n)",  # "I'll calculate..."
            r"Let me.*?(?=Equipment|\n\n)",  # "Let me break down..."
            r"^\*\*Capacity.*?\n",  # Lines starting with **Capacity
            r"^Input.*?:\s*\d+.*?\n",  # "Input Capacity: X"
        ]
        
        cleaned = output
        
        # Remove unwanted patterns
        for pattern in unwanted_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
        
        # Remove multiple blank lines
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        # Remove leading/trailing whitespace
        cleaned = cleaned.strip()
        
        # If the output starts with text before "Equipment", remove it
        lines = cleaned.split('\n')
        start_idx = 0
        
        for i, line in enumerate(lines):
            # Look for the first line that contains "Equipment" or starts with "---" or "Total"
            if (re.search(r'Equipment\s+\d+', line, re.IGNORECASE) or 
                line.strip().startswith('---') or
                re.search(r'^Total\s+Cost', line, re.IGNORECASE)):
                start_idx = i
                break
        
        # Reconstruct from the first relevant line
        if start_idx > 0:
            cleaned = '\n'.join(lines[start_idx:])
        
        return cleaned.strip()
    
    def generate(self, type : str, input: int) -> str:
        print("Generating response")
        prompt = self.build_prompt(type, input)
        print(f"Prompt: {prompt}")
        response = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
        
        # Clean the output before returning
        raw_output = response.message.content
        cleaned_output = self.clean_output(raw_output)
        
        return cleaned_output

# Example usage:
# classifier = OllamaClassifier(model_name="your-model-name")
# result = classifier.generate(type="KLT", input=100)