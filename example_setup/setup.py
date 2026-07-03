'''Example of an experiment setup class. Modify to match your actual hardware and experiment requirements.'''
import os
import sys

from safe_lab_agents import quantity  # Wrapper for physical quantities with units enables improved logging

sys.path.append(os.path.dirname(__file__))  # Add the current directory to the Python path
from simulation import simulate_experiment


class ExampleOpticalSetup:
    """Class representing an optical experiment setup:
        1. Laser source
        2. Horizontal polarizer
        3. Rotatable Lambda/quarter waveplate
        4. Rotatable Lambda/half waveplate
        5. Rotatable polarizer
        6. Detector to measure the intensity of light
    """

    def __init__(self, initial_polarizer_angle: float = 0.0, 
                 initial_lambda_quarter_angle: float = 0.0, 
                 initial_lambda_half_angle: float = 0.0):
        """Initialize the experimental setup, including hardware connections and default parameters."""
        self.polarizer_angle = initial_polarizer_angle
        self.lambda_quarter_angle = initial_lambda_quarter_angle
        self.lambda_half_angle = initial_lambda_half_angle
        print("Setup: initialized optical setup with default angles.")
    
    
    # Use type hints and clear docstrings. They will be used by the agent to understand the function's purpose and how to call it.
    def set_angle(self, angle: float, component: str) -> str:
        """Set the angle of the optical component.
        Angles are not absolute but only up to a fixed, unknown offset.

        Args:
            angle: Desired angle in degrees (0 to 360).
            component: The optical component whose angle to set ('polarizer', 'lambda_quarter', or 'lambda_half').
        Returns:
            A string indicating success.
        Raises:
            ValueError: If the angle is out of the valid range or if the component name is invalid.
        """

        # Hard enforce safety checks in the functions provided to the agent. The agent can only call these functions, so it cannot bypass these checks.
        if not 0.0 <= angle <= 360.0:
            raise ValueError(f"Angle {angle} out of range [0, 360].")
    
        if component == "polarizer":
            self.polarizer_angle = angle
            # Replace with your actual instrument code to set the polarizer angle
        elif component == "lambda_quarter":
            self.lambda_quarter_angle = angle
            # Replace with your actual instrument code to set the lambda/quarter waveplate angle
        elif component == "lambda_half":
            self.lambda_half_angle = angle
            # Replace with your actual instrument code to set the lambda/half waveplate angle
        else:
            raise ValueError(f"Invalid component: {component}. Use 'polarizer', 'lambda_quarter', or 'lambda_half'.")
        
        return f"{component.capitalize()} angle set to {angle} degrees."
    
    def measure_power(self) -> dict:
        """Measure the optical power at the detector.

        Returns:
            A dictionary containing the measured optical power in Watts.
        """
        # Simulation of the setup. Replace with your actual instrument code to measure power
        power = simulate_experiment(
            self.polarizer_angle, self.lambda_quarter_angle, self.lambda_half_angle
        )

        # Recommended format for returning measurement results:
        # Use a dictionary with keys as measurement names and values as quantities with units.
        # This will make the auto-log functionality of the agent work best.
        return {'power': quantity(power, "W")}
        
    def close(self) -> None:
        """Release the hardware connection."""
        # Replace with your actual instrument code to close the connection to the setup
        print("Setup: closed connection to the optical setup.")
    

            
