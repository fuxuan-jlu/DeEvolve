from aspe_interface import ASPE_Interface
import os

def main():
    # Define all parameters in a dictionary
    paras = {
        'size': 5,  # Population size
        "api_endpoint": "xxxxx",   #Your api_endpoint
        "api_key": "xxxxx",  #Your api_key
        "model_LLM": "xxxxx",  #Your model type
        'debug_mode': False,  # Set to False in production
        'generations': 5,  # Number of generations to evolve
        'max_pop_size': 10  # Maximum population size during evolution
    }

    # Initialize ASPE interface with parameters
    aspe = ASPE_Interface(
        size=paras['size'],
        api_endpoint=paras['api_endpoint'],
        api_key=paras['api_key'],
        model_LLM=paras['model_LLM'],
        debug_mode=paras['debug_mode']
    )

    # Run the evolutionary process
    aspe.evolve(m=paras['generations'], maxsize=paras['max_pop_size'])

    # Get and display the best solution
    best_solution = aspe.get_best_solution()
    print("\nBest Solution Found:")
    print(f"Score: {best_solution['score']}")
    print(f"Code:\n{best_solution['code']}")


if __name__ == "__main__":
    main()
