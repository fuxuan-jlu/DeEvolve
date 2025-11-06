import sys
from copy import deepcopy

from prompts import Prompts
from evaluate import Evaluate
import code_init
import json
import ast
import re
import traceback
import random
import time
from io import StringIO
import contextlib

class ASPE_Interface:
    def __init__(self, size, api_endpoint, api_key, model_LLM, debug_mode):
        self.size = size  # Population size
        self.prompts = Prompts(api_endpoint, api_key, model_LLM, debug_mode)
        self.population = []  # Will store dictionaries with program_info
        self.generation = 0
        self.code_init = code_init.CodeInit()  # Initialize code_init instance

    def initialize_population(self):
        """Initialize population using the multi-agent system to mutate the initial code"""
        initial_code_1 = self.code_init.get_initial_code_1()  # Get initial code from code_init instance
        program_info_1 = {'code': initial_code_1}
        initial_code_2 = self.code_init.get_initial_code_2()  # Get initial code from code_init instance
        program_info_2 = {'code': initial_code_2}
        run_time, completion_time = Evaluate(program_info_1['code'])
        program_info_1['time'] = run_time
        program_info_1['score'] = completion_time
        run_time, completion_time = Evaluate(program_info_2['code'])
        program_info_2['time'] = run_time
        program_info_2['score'] = completion_time
        print(program_info_1['score'])
        print(program_info_2['score'])
        self.population.append(program_info_1)
        self.population.append(program_info_2)
        """
        for _ in range(self.size):
            # Initialize program_info with the base code
            # Use the multi-agent system to generate a variant of the initial code
            new_individual = self.generate_new_individual(program_info)
            self.population.append(new_individual)
        """

    def generate_new_individual(self, pop, operator):
        """Use the multi-agent system to generate a new individual"""
        program_info = {
            'code': None,
            'new_algorithm': None,
        }
        try:
            if operator == 'e1':
                parents = random.sample(pop, 2)
                response = self.prompts.get_alg_imp_1(parents[0],parents[1])
                #print(response)
                program_info.update(self.prompts.extract_new_algorithm(response))
                program_info.update({'code': parents[0]['code']})
                response_code = self.prompts.get_code_imp("FSTSP", program_info)
            elif operator == 'e2':
                parents = random.sample(pop, 2)
                response = self.prompts.get_alg_imp_2(parents[0],parents[1])
                program_info.update(self.prompts.extract_new_algorithm(response))
                program_info.update({'code': parents[0]['code']})
                response_code = self.prompts.get_code_imp("FSTSP", program_info)
            elif operator == 'm1':
                parent = random.choice(pop)
                response = self.prompts.get_alg_imp_3(parent)
                program_info.update(self.prompts.extract_new_algorithm(response))
                program_info.update({'code': parent['code']})
                response_code = self.prompts.get_code_imp("FSTSP", program_info)
            elif operator == 'm2':
                parent = random.choice(pop)
                response = self.prompts.get_alg_imp_4(parent)
                program_info.update(self.prompts.extract_new_algorithm(response))
                program_info.update({'code': parent['code']})
                response_code = self.prompts.get_code_imp("FSTSP", program_info)
            elif operator == 'm3':
                parent = random.choice(pop)
                response = self.prompts.get_alg_imp_5(parent)
                program_info.update(self.prompts.extract_code(response))
                run_time, completion_time = Evaluate(program_info['code'])
                program_info['time'] = run_time
                program_info['score'] = completion_time
                return program_info
            # Step 4: Code improvement
            print("yes!")
            program_info.update(self.prompts.extract_code(response_code))
            # print(program_info)
            run_time, completion_time = Evaluate(program_info['code'])
            program_info['time'] = run_time
            program_info['score'] = completion_time
            return program_info


        except Exception as e:
            m = 0
            # If there's an error, use the debug agent to fix the code
            while m < 5:
                try:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    # Get full stack trace as string
                    full_traceback = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                    print(f"Error in generation process: {full_traceback}")
                    program_info['error'] = full_traceback  # Store complete traceback instead of just error message
                    #print(f"Error in generation process: {str(e)}")
                    #program_info['error'] = str(e)
                    response_debug = self.prompts.get_code_debug("FSTSP", program_info)
                    program_info.update(self.prompts.extract_code(response_debug))
                    run_time, completion_time = Evaluate(program_info['code'])
                    program_info['time'] = run_time
                    program_info['score'] = completion_time
                    return program_info
                except Exception as e:
                    m += 1
            program_info['score'] = 1500
            program_info['time'] = 1500
            return program_info

    def evolve(self, m, operators):
        """Run the evolutionary process for m generations"""
        if not self.population:
            self.initialize_population()

        for generation in range(m):
            self.generation += 1
            for op in operators:
                offspring = self.generate_new_individual(self.population,op)
            # Generate new individuals
                #print("yes!")
                print(offspring)
                self.population.append(offspring)

            # Sort and select the best individuals
            self.population.sort(key=lambda x: x['score'], reverse=False)
            self.population = self.population[:self.size]

            print(f"Generation {self.generation}: Best score = {self.population[0]['score']}")

    def get_best_solution(self):
        """Return the best solution found"""
        if not self.population:
            return None
        self.population.sort(key=lambda x: x['score'], reverse=False)
        return self.population[0]

