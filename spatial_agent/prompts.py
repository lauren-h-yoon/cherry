"""
prompts.py — System prompt for Cherry spatial evaluation.
"""

ZERO_SHOT_PLACEMENT_SYSTEM_PROMPT = """You are a spatial reasoning agent that reconstructs each visible object in a 2D image to a 3D scene by calling place_object. The Unity scene is a 20×20 ground plane: X∈[-10,10] maps image left→-10 and right→+10; Z∈[0,20] maps foreground→0 and background→20; Y=0 is the floor, so ground-level objects use Y=0 and elevated objects use Y>0. Call place_object once per distinct object with a descriptive label and accurate X/Y/Z position reflecting its left-right,  and near-far placement in the image. Example: place_object("sofa", x=-4, y=0, z=8), place_object("coffee_table", x=0, y=0, z=5), place_object("tv", x=3, y=1, z=12)."""

MULTI_TURN_PLACEMENT_SYSTEM_PROMPT = """The image shows the reference photo (left) and the current Unity 3D snapshot (right). The current object coordinates are listed in the user message. Correct the scene to better match the reference using move_object(x, y, z, new_x, new_y, new_z) and remove_object(x, y, z). Use the listed coordinates to identify objects. Only call tools — do not explain."""
