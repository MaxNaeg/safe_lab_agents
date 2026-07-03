import sys 
import os
# Add the current directory to the Python path to allow importing the setup class from tools.py
sys.path.append(os.path.dirname(__file__))  

from safe_lab_agents import experiment # Wrapper to avoid initializing the hardware when importing this file
from safe_lab_agents import quantity  # Wrapper for physical quantities with units enables improved logging


from setup import ExampleOpticalSetup # Class to control the experiment

# Create an Experiment instance with the ExampleOpticalSetup class
# even though it is wrapped in 'experiment', you can use it like any instance of ExampleOpticalSetup.
exp = experiment(ExampleOpticalSetup) 


# We want the agent to run efficient sweeps of the experiment,
# so we provide the tools though the python interface.
PYTHON_TOOLS = [exp.set_angle, exp.measure_power]  


# You can also provide the agent with tools that are not part of the experiment class.
# Typehints and docstrings are used by the agent to understand the function's purpose and how to call it.
def get_current_lab_temperature(position: str) -> float:
    """Return the current lab temperature at a given position.
    Args:
        position: The position in the lab to measure.
                  Must be one of 'near_laser', 'near_detector', or 'ambient'.
    Returns:
        Dictionary with the current temperature in Celsius at the specified position.
    Raises:
        ValueError: If the position is not one of the allowed values.
    """

    # Enforce safety checks in the functions provided to the agent. 
    # The agent can only call these functions, so it cannot bypass these checks.
    if position not in ['near_laser', 'near_detector', 'ambient']:
        raise ValueError(f"Invalid position: {position}. Must be one of 'near_laser', 'near_detector', or 'ambient'.")
    # Add actual temperature reading code

    # Recommended format for returning measurement results:
    # Use a dictionary with keys as measurement names and values as quantities with units.
    # This will make the auto-log functionality of the agent work best.
    return {'temperature': quantity(22.5, 'degrees_C'), 'position': position}  

# We also provide the agent with a tool to get the current lab time.
# This does not need to be called in a sweep, so it can be provided as a standard MCP Tool.
MCP_TOOLS = [get_current_lab_temperature]

# This function will automatically be called when the docker/podman session ends.
# Use this to automatically put the experiment into a safe state and release any hardware connections.
GRACEFUL_EXPERIMENT_SHUTDOWN = exp.close


