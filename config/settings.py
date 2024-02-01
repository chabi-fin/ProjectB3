# settings.py

# General settings
PROJECT_NAME = "ProjectB3"
DEBUG_MODE = True

# Path settings
path_head = "/home/lf1071fu/project_b3"
data_head = f"{ path_head }/simulate"
figure_head = f"{ path_head }/figures"
struct_head = f"{ path_head }/structures"

# Protein settings related to residue IDs
# Shift residue count for consistency with the full-toxin
toxin_resid = 544 
beta_flap_group = "backbone and resid 195-218"
alpha_flap_group = "backbone and resid 219-232"
active_res_group = "resid 110 or resid 155"

# Aesthetics
open_color = "#A1DEA1"
closed_color = "#EAAFCC"