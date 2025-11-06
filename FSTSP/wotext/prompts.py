import ast
import json
import re
from interface_llm import InterfaceLLM

class Prompts:
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode):
        self.interface_llm = InterfaceLLM(
            api_endpoint, api_key,
            model_LLM, debug_mode
        )

    def get_sub_prob(self, prob_name, program_info):
        """改进后的提示词"""
        prompt = f"""
        You MUST respond in valid JSON format ONLY. Example:
           [
                {{
                    "Sub_problem": "description", 
                    "Solving_Algorithm": "algorithm"
                }} , ...
            ]

        Analyze this {prob_name} solution code:
        {program_info['code']}

        Identify sub-problems and their algorithms. Return ONLY the JSON object.
        """
        return self.interface_llm.get_response(prompt)

    def get_select_prob(self, prob_name, program_info):
        """Sub-problem selection prompt"""
        prompt = f"""
        code:
        {program_info['code']}
        The above is a solution program for an {prob_name} problem. The program solves the following sub-problems: {program_info['sub_prob_alg']}
        Please choose the sub-problem whose solving algorithm needs the most improvement. By improving its algorithm, the program's running efficiency and solution quality can be maximized.
        Return in the form: {{"..."}}
        The part "..." must be a nature language description of the sub-problem
        Do not make extra explanation.
        """
        return self.interface_llm.get_response(prompt)

    def get_alg_ref(self, prob_name, program_info):
        """Algorithm improvement prompt"""
        prompt = f"""
        code:
        {program_info['code']}
        The above is a solution program for an {prob_name} problem.
        The program solves the following sub-problems: {program_info['sub_prob_alg']}
        Please reflect on the shortcomings of the algorithm corresponding to the sub-problem {program_info['sub_problem']}.
        Return in the form: {{"..."}}
        The part "..." must be a nature language description of the reflection
        Do not make extra explanation.
        """
        return self.interface_llm.get_response(prompt)

    def get_alg_imp(self, prob_name, program_info):
        """Algorithm improvement prompt"""
        prompt = f"""
        code:
        {program_info['code']}
        The above is a solution program for an {prob_name} problem.
        The program solves the following sub-problems: {program_info['sub_prob_alg']}
        The reflection of the solution algorithm of the sub-problem: {program_info['sub_problem']} is:
        {program_info['reflection']}
        Please generate a new algorithm for the sub-problem based on this reflection.
        Return in the form: {{"..."}}
        The part "..." must be a nature language description of new algorithm
        Do not make extra explanation.
        """
        return self.interface_llm.get_response(prompt)

    def get_code_imp(self, prob_name, program_info):
        """Code update prompt"""
        #No escape symbols are allowed in the code.
        prompt = f"""
        code:
        {program_info['code']}
        The above is a solution program for an {prob_name} problem. The program solves the following sub-problems: {program_info['sub_prob_alg']}
        Among them, for {program_info['sub_problem']}, 
        First,generate the new algorithm for this sub problem
        Next, generate the new complete improved program.
        Ensure the code includes ALL necessary imports.
        Return in the form: {{"..."}}
        The part "..." must be the complete improved program
        Do not make extra explanation.
        """
        return self.interface_llm.get_response(prompt)

    def get_code_debug(self, prob_name, program_info):
        """Code debugging prompt"""
        prompt = f"""
        code:
        {program_info['code']}
        The above is a solution program for an {prob_name} problem.
        The program encountered the following error: {program_info['error']}.
        Modify the code based on the error message and provide the corrected complete code.
        Return in the form: {{"..."}}
        The part "..." must be the corrected complete program
        Do not make extra explanation.
        """
        return self.interface_llm.get_response(prompt)

    def extract_sub_problems(self, response):
        """Extract all sub-problems and their corresponding solving algorithms"""
        if not isinstance(response, str):
            print(f"Error: response must be a string, got {type(response)}")
            return {'sub_prob_alg': []}

        # Remove Markdown code block markers if present
        cleaned_response = response.strip()
        if cleaned_response.startswith('```json') and cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[7:-3].strip()  # Remove ```json and ```
        elif cleaned_response.startswith('```') and cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[3:-3].strip()  # Remove generic ```

        try:
            data = json.loads(cleaned_response)

            if not isinstance(data, list):
                print("Error: Expected JSON array, got", type(data))
                return {'sub_prob_alg': []}

            # Validate and extract each item
            sub_prob_alg = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                sub_problem = item.get("Sub_problem")
                algorithm = item.get("Solving_Algorithm")
                if sub_problem and algorithm:
                    sub_prob_alg.append({
                        "Sub_problem": sub_problem,
                        "Solving_Algorithm": algorithm
                    })

            return {'sub_prob_alg': sub_prob_alg}

        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}\nResponse content:\n{cleaned_response}")
            return {'sub_prob_alg': []}
        except Exception as e:
            print(f"Unexpected Error: {e}")
            return {'sub_prob_alg': []}

    def extract_sub_problem(self, response):
        """Extract selected sub-problem from a JSON response."""
        try:

            match = re.search(r'\{.*?\}', response, re.DOTALL)
            if match:
                return {'sub_problem': match.group(0)}
            else:
                return None
        except Exception as e:
            print(f"Error extracting sub-problem: {e}")
            return None

    def extract_reflection(self, response):
        """Extract new algorithm into program_info"""
        try:
            match = re.search(r'\{.*?\}', response, re.DOTALL)
            if match:
                try:
                    return {'reflection':match.group(0)}
                except json.JSONDecodeError as je:
                    print(f"Invalid JSON: {je}")
                    return None
            else:
                return None
        except Exception as e:
            print(f"Error extracting reflection: {e}")
            return None

    def extract_new_algorithm(self, response):
        """Extract new algorithm into program_info"""
        try:
            match = re.search(r'\{.*?\}', response, re.DOTALL)
            if match:
                try:
                    return {'new_algorithm':match.group(0)}
                except json.JSONDecodeError as je:
                    print(f"Invalid JSON: {je}")
                    return None
            else:
                return None
        except Exception as e:
            print(f"Error extracting new algorithm: {e}")
            return None
    def extract_code(self, response):
        """Extract code from responses that contain JSON code blocks"""

        try:
            match = re.search(r'{(.*)}', response, re.DOTALL)
            if match:
                print("good!")
                pos = match.group(1).find('import')
                match = match.group(1)[pos:]
                return {'code': match}
            else:
                match = re.search(r'(import.*[\'"]$)', response, re.DOTALL)
                if match:
                    return {'code': match.group(0)}
                else:
                    print("No matching pattern was found in the response")
                    return {'code': ''}  # If there is no match, return an empty code dictionary
        except Exception as e:
            print(f"error: {e}")
            return {'code': ''}

