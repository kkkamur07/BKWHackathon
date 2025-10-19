## Assuming Expected output to be KLT : KW ; HLT : KW and RLT : m^3/h

import ollama
import pandas as pd


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
    
    def generate(self, type : str, input: int) -> str:
        print("Generating response")
        prompt = self.build_prompt(type, input)
        print(f"Prompt: {prompt}")
        response = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
        return response.message.content

# Example usage:
# classifier = OllamaClassifier(model_name="your-model-name")
# result = classifier.generate(type="KLT", input=100)

    