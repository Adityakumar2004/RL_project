# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom Keyboard controller with additional functionality."""

import numpy as np
import weakref
from collections.abc import Callable
from scipy.spatial.transform import Rotation

import carb
import omni

from isaaclab.devices.device_base import DeviceBase


class CustomKeyboard(DeviceBase):
    """A custom keyboard controller with additional functionality for recording, friction control, etc.
    
    Extends the basic SE(3) control with additional keys for:
    - Recording start/stop/pause
    - Dynamic friction adjustment
    - Parameter experimentation
    - Custom user-defined actions
    
    Key bindings:
        ============================== ================= =================
        Description                    Key               Action
        ============================== ================= =================
        Basic SE(3) Movement:
        Toggle gripper (open/close)    K                 
        Move along x-axis              W/S               +x/-x
        Move along y-axis              A/D               +y/-y
        Move along z-axis              Q/E               +z/-z
        Rotate along x-axis            Z/X               +roll/-roll
        Rotate along y-axis            T/G               +pitch/-pitch
        Rotate along z-axis            C/V               +yaw/-yaw
        Reset pose                     L                 
        
        Additional Custom Functions:
        Start recording                R                 
        Stop recording                 F                 
        Pause/Resume recording         P                 
        Increase friction              I                 
        Decrease friction              O                 
        Reset friction                 U                 
        Save current state             1                 
        Load saved state               2                 
        Toggle debug mode              3                 
        Emergency stop                 SPACE             
        ============================== ================= =================
    """

    def __init__(self, pos_sensitivity: float = 0.4, rot_sensitivity: float = 0.8, 
                 friction_step: float = 0.1):
        """Initialize the custom keyboard controller.

        Args:
            pos_sensitivity: Magnitude of input position command scaling.
            rot_sensitivity: Magnitude of scale input rotation commands scaling.
            friction_step: Step size for friction adjustments.
        """
        # Call parent constructor
        super().__init__()
        
        # Store inputs
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.friction_step = friction_step
        
        # Acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        
        # Subscribe to keyboard events
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        
        # Create key bindings
        self._create_key_bindings()
        
        # Command buffers for SE(3) control
        self._close_gripper = False
        self._delta_pos = np.zeros(3)  # (x, y, z)
        self._delta_rot = np.zeros(3)  # (roll, pitch, yaw)
        
        # Additional state variables
        self._recording_state = "stopped"  # "stopped", "recording", "paused"
        self._current_friction = 1.0
        self._debug_mode = False
        self._emergency_stop = False
        self._saved_state = None
        
        # Dictionary for additional callbacks
        self._additional_callbacks = dict()
        
        # Dictionary to track key states (for debugging)
        self._key_states = {}

    def __del__(self):
        """Release the keyboard interface."""
        if hasattr(self, '_keyboard_sub') and self._keyboard_sub is not None:
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of the custom keyboard controller."""
        msg = f"Custom Keyboard Controller: {self.__class__.__name__}\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tSE(3) Control:\n"
        msg += "\t  Toggle gripper: K\n"
        msg += "\t  Move arm: W/S (x), A/D (y), Q/E (z)\n"
        msg += "\t  Rotate arm: Z/X (roll), T/G (pitch), C/V (yaw)\n"
        msg += "\t  Reset pose: L\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tAdditional Functions:\n"
        msg += "\t  Recording: R (start), F (stop), P (pause/resume)\n"
        msg += "\t  Friction: I (increase), O (decrease), U (reset)\n"
        msg += "\t  States: 1 (save), 2 (load), 3 (debug toggle)\n"
        msg += "\t  Emergency stop: SPACE\n"
        msg += "\t----------------------------------------------\n"
        msg += f"\t  Current friction: {self._current_friction:.2f}\n"
        msg += f"\t  Recording state: {self._recording_state}\n"
        msg += f"\t  Debug mode: {self._debug_mode}\n"
        return msg

    """
    Operations
    """

    def reset(self):
        """Reset the internals."""
        # Reset SE(3) command buffers
        self._close_gripper = False
        self._delta_pos = np.zeros(3)
        self._delta_rot = np.zeros(3)
        
        # Reset additional states (optional - you might want to keep some states)
        self._emergency_stop = False
        print("Controller reset!")

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind keyboard.

        Args:
            key: The keyboard button to check against.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        self._additional_callbacks[key] = func
        print(f"Added callback for key: {key}")

    def advance(self) -> dict:
        """Provides the result from keyboard event state.

        Returns:
            A dictionary containing all the current states and commands.
        """
        # Convert to rotation vector
        rot_vec = Rotation.from_euler("XYZ", self._delta_rot).as_rotvec()
        
        # Create comprehensive output dictionary
        output = {
            # SE(3) control
            "pose_command": np.concatenate([self._delta_pos, rot_vec]),
            "gripper_command": self._close_gripper,
            
            # Additional states
            "recording_state": self._recording_state,
            "current_friction": self._current_friction,
            "debug_mode": self._debug_mode,
            "emergency_stop": self._emergency_stop,
            
            # Raw values for debugging
            "delta_pos": self._delta_pos.copy(),
            "delta_rot": self._delta_rot.copy(),
        }
        
        return output

    # Getter methods for individual states
    def get_pose_command(self) -> tuple[np.ndarray, bool]:
        """Get just the SE(3) command (compatible with original Se3Keyboard)."""
        rot_vec = Rotation.from_euler("XYZ", self._delta_rot).as_rotvec()
        return np.concatenate([self._delta_pos, rot_vec]), self._close_gripper

    def get_recording_state(self) -> str:
        """Get current recording state."""
        return self._recording_state

    def get_friction(self) -> float:
        """Get current friction value."""
        return self._current_friction

    def is_emergency_stop(self) -> bool:
        """Check if emergency stop is activated."""
        return self._emergency_stop

    def is_debug_mode(self) -> bool:
        """Check if debug mode is active."""
        return self._debug_mode

    """
    Internal helpers.
    """

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Subscriber callback for keyboard events."""
        
        # Store key state for debugging
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._key_states[event.input.name] = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._key_states[event.input.name] = False

        # Handle key press events
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._handle_key_press(event.input.name)
        
        # Handle key release events
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._handle_key_release(event.input.name)

        return True

    def _handle_key_press(self, key_name: str):
        """Handle key press events."""
        
        # Original SE(3) controls - single press actions
        if key_name == "L":
            self.reset()
        elif key_name == "K":
            self._close_gripper = not self._close_gripper
            print(f"Gripper: {'CLOSE' if self._close_gripper else 'OPEN'}")
        
        # Movement controls - continuous actions (add on press)
        elif key_name in ["W", "S", "A", "D", "Q", "E"]:
            self._delta_pos += self._INPUT_KEY_MAPPING[key_name]
        elif key_name in ["Z", "X", "T", "G", "C", "V"]:
            self._delta_rot += self._INPUT_KEY_MAPPING[key_name]
        
        # Custom functionality - Recording controls
        elif key_name == "R":
            self._start_recording()
        elif key_name == "F":
            self._stop_recording()
        elif key_name == "P":
            self._toggle_pause_recording()
        
        # Friction controls
        elif key_name == "I":
            self._increase_friction()
        elif key_name == "O":
            self._decrease_friction()
        elif key_name == "U":
            self._reset_friction()
        
        # State management
        elif key_name == "1":
            self._save_state()
        elif key_name == "2":
            self._load_state()
        elif key_name == "3":
            self._toggle_debug_mode()
        
        # Emergency stop
        elif key_name == "SPACE":
            self._emergency_stop = not self._emergency_stop
            print(f"EMERGENCY STOP: {'ACTIVATED' if self._emergency_stop else 'DEACTIVATED'}")
        
        # Additional callbacks
        elif key_name in self._additional_callbacks:
            self._additional_callbacks[key_name]()

    def _handle_key_release(self, key_name: str):
        """Handle key release events."""
        
        # Movement controls - continuous actions (subtract on release)
        if key_name in ["W", "S", "A", "D", "Q", "E"]:
            self._delta_pos -= self._INPUT_KEY_MAPPING[key_name]
        elif key_name in ["Z", "X", "T", "G", "C", "V"]:
            self._delta_rot -= self._INPUT_KEY_MAPPING[key_name]

    def _create_key_bindings(self):
        """Creates key bindings mapping."""
        self._INPUT_KEY_MAPPING = {
            # SE(3) movement mappings
            # x-axis (forward/backward)
            "W": np.asarray([1.0, 0.0, 0.0]) * self.pos_sensitivity,
            "S": np.asarray([-1.0, 0.0, 0.0]) * self.pos_sensitivity,
            # y-axis (left/right)
            "A": np.asarray([0.0, 1.0, 0.0]) * self.pos_sensitivity,
            "D": np.asarray([0.0, -1.0, 0.0]) * self.pos_sensitivity,
            # z-axis (up/down)
            "Q": np.asarray([0.0, 0.0, 1.0]) * self.pos_sensitivity,
            "E": np.asarray([0.0, 0.0, -1.0]) * self.pos_sensitivity,
            # roll (around x-axis)
            "Z": np.asarray([1.0, 0.0, 0.0]) * self.rot_sensitivity,
            "X": np.asarray([-1.0, 0.0, 0.0]) * self.rot_sensitivity,
            # pitch (around y-axis)
            "T": np.asarray([0.0, 1.0, 0.0]) * self.rot_sensitivity,
            "G": np.asarray([0.0, -1.0, 0.0]) * self.rot_sensitivity,
            # yaw (around z-axis)
            "C": np.asarray([0.0, 0.0, 1.0]) * self.rot_sensitivity,
            "V": np.asarray([0.0, 0.0, -1.0]) * self.rot_sensitivity,
        }

    # Custom functionality methods
    def _start_recording(self):
        """Start recording."""
        self._recording_state = "recording"
        print("Recording STARTED")

    def _stop_recording(self):
        """Stop recording."""
        self._recording_state = "stopped"
        print("Recording STOPPED")

    def _toggle_pause_recording(self):
        """Toggle pause/resume recording."""
        if self._recording_state == "recording":
            self._recording_state = "paused"
            print("Recording PAUSED")
        elif self._recording_state == "paused":
            self._recording_state = "recording"
            print("Recording RESUMED")
        else:
            print("Cannot pause - recording not active")

    def _increase_friction(self):
        """Increase friction value."""
        self._current_friction = min(2.0, self._current_friction + self.friction_step)
        print(f"Friction increased to: {self._current_friction:.2f}")

    def _decrease_friction(self):
        """Decrease friction value."""
        self._current_friction = max(0.0, self._current_friction - self.friction_step)
        print(f"Friction decreased to: {self._current_friction:.2f}")

    def _reset_friction(self):
        """Reset friction to default value."""
        self._current_friction = 1.0
        print(f"Friction reset to: {self._current_friction:.2f}")

    def _save_state(self):
        """Save current state."""
        self._saved_state = {
            "friction": self._current_friction,
            "delta_pos": self._delta_pos.copy(),
            "delta_rot": self._delta_rot.copy(),
            "gripper": self._close_gripper,
        }
        print("State SAVED")

    def _load_state(self):
        """Load saved state."""
        if self._saved_state is not None:
            self._current_friction = self._saved_state["friction"]
            self._delta_pos = self._saved_state["delta_pos"].copy()
            self._delta_rot = self._saved_state["delta_rot"].copy()
            self._close_gripper = self._saved_state["gripper"]
            print("State LOADED")
        else:
            print("No saved state available")

    def _toggle_debug_mode(self):
        """Toggle debug mode."""
        self._debug_mode = not self._debug_mode
        print(f"Debug mode: {'ON' if self._debug_mode else 'OFF'}")
        
        if self._debug_mode:
            self._print_debug_info()

    def _print_debug_info(self):
        """Print current debug information."""
        print("--- DEBUG INFO ---")
        print(f"Position delta: {self._delta_pos}")
        print(f"Rotation delta: {self._delta_rot}")
        print(f"Gripper: {'CLOSE' if self._close_gripper else 'OPEN'}")
        print(f"Friction: {self._current_friction}")
        print(f"Recording: {self._recording_state}")
        print(f"Currently pressed keys: {[k for k, v in self._key_states.items() if v]}")
        print("------------------")
