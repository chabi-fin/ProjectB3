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
alpha_flap_group = "backbone and resid 219-233"
active_res_group = "resid 110 or resid 155"

# Alpha carbons for the beta-flap vector 
r1, r2 = 206, 215

# Aesthetics
open_color = "#A1DEA1"
closed_color = "#EAAFCC"
styles = {"apo-open" : ("#2E7D32", "solid", "X"), 
          "apo-closed" : ("#1976D2", "dashed", "X"),
          "holo-open" : ("#FF6F00", "solid", "o"), 
          "holo-closed" : ("#8E24AA", "dashed", "o"),
          "K57G" : ("#00897B", "dotted", "P"),
          "E200G" : ("#FF6F00", "dotted", "P"),
          "double-mut" : ("#5E35B1", "dotted", "P")}

# Critical contacts dictionary
selections = {
    "L212--E200" : ("resid 212 and name N","resid 200 and name O"),
    "K249--K232" : ("resid 249 and name NZ","resid 232 and name O"),
    "K249--S230a" : ("resid 249 and name N","resid 230 and name OG"),
    "K249--S230b" : ("resid 249 and name N","resid 230 and name O"),
    "K249--S230c" : ("resid 249 and name NZ","resid 230 and name O"),
    "S230--I226" : ("resid 230 and name N","resid 226 and name O"),
    "K249--E233" : ("resid 249 and name NZ","resid 233 and name OE*"),
    "R209--S231" : ("resid 209 and name NH*",
        "resid 231 and (name OG or name O)"),
    "R209--N197" : ("resid 209 and name NH*","resid 197 and name OD*"),
    "R209--E253" : ("resid 209 and name NH*","resid 253 and name OE*"),
    "R202--E210" : ("resid 202 and name NE","resid 210 and name OE*"),
    "K221--E223" : ("resid 221 and name NZ","resid 223 and name OE*"),
    "R208--E222" : ("resid 208 and name NH*","resid 222 and name OE*"),
    "K57--G207" : ("resid 57 and name NZ","resid 207 and name O"),
    "K57--V201" : ("resid 57 and name NZ","resid 201 and name O"),
    "R28--S205" : ("resid 28 and name NH*","resid 205 and name O"),
    "N204--R208" : ("resid 204 and (name N or name O)",
        "resid 208 and (name N or name O)"),
    "E210--N204" : ("resid 204 and name ND2", "resid 210 and name OE*"),
    "R202--E210" : ("resid 202 and (name NE* or name NH*)",
        "resid 210 and name OE*"),
    "R202--W218" : ("resid 202 and (name NE* or name NH*)",
        "resid 218 and name CZ*"),
    "I219--E223" : ("resid 219 and name CA", "resid 223 and name CA"),
    "Y199--D228" : ("resid 199 and name OH", "resid 228 and name OD*"),
    "I219--E223" : ("resid 219 and (name CD or name CG*)", 
        "resid 223 and name OE*"),
    "L212--W218" : ("resid 218 and name CA", "resid 212 and name CA"),
    "E206--N204" : ("resid 206 and name N", "resid 204 and name OD1"),
    "C155--H214" : ("resid 155 and name SG", 
        "resid 214 and (name ND1 or name NE2)"),
    "C155--H110" : ("resid 155 and name SG", 
        "resid 110 and (name ND1 or name NE2)"),
    # Proposed allosteric network
    "K57--E200" : ("resid 57 and name NZ*","resid 200 and name OE*"),
    "E200--N53" : ("resid 53 and name ND2","resid 200 and name OE*"),
    "N53--E49" : ("resid 49 and name OE*", "resid 53 and name ND2"),
    "E49--W128" : ("resid 218 and name NE1", "resid 49 and name OE*"),
    # Binding pocket contacts
    "K221--IP6" : ("resid 221 and name NZ","resname IPL and name O*"),
    "R208--IP6" : ("resid 208 and name N*","resname IPL and name O*"),
    "R28--IP6" : ("resid 28 and name NH*","resname IPL and name O*"),
    "R32--IP6" : ("resid 32 and name NH*","resname IPL and name O*"),
    "Y34--IP6" : ("resid 34 and name OH","resname IPL and name O*"),
    "K249--IP6" : ("resid 249 and name NZ","resname IPL and name O*"),
    "R209--IP6" : ("resid 209 and name NH*","resname IPL and name O*"),
    "K104--IP6" : ("resid 104 and name NZ","resname IPL and name O*"),
    "K57--IP6" : ("resid 57 and name NZ","resname IPL and name O*"),
    "K232--IP6" : ("resid 232 and name NZ","resname IPL and name O*"),
    "Y234--IP6" : ("resid 234 and name OH","resname IPL and name O*"),
    "K102--IP6" : ("resid 102 and name NZ","resname IPL and name O*"),
    # Alpha-flap
    "N220--S224" : ("resid 220 and name O", "resid 224 and name N"),
    "K221--I225" : ("resid 221 and name O", "resid 225 and name N"),
    "E222--I226" : ("resid 222 and name O", "resid 226 and name N"),
    "E223--K227" : ("resid 223 and name O", "resid 227 and name N"),
    "S224--D228" : ("resid 224 and name O", "resid 228 and name N"),
    "I225--I229" : ("resid 225 and name O", "resid 229 and name N"),
    "I226--S230" : ("resid 226 and name O", "resid 230 and name N"),
    "K227--S231" : ("resid 227 and name O", "resid 231 and name N"),
    "D228--K232" : ("resid 228 and name O", "resid 232 and name N"),
    "I229--E233" : ("resid 229 and name O", "resid 233 and name N"),
    "S224--I229" : ("resid 224 and name O", "resid 229 and name N"),
    # Does this network change?
    "K57--E206" : ("resid 206 and name O*", "resid 57 and name NZ"), # backbone or sidechain of E206
    "E21--K22" : ("resid 21 and name OE*","resid 22 and name NZ*"),
    "K22--E49" : ("resid 21 and name OE*","resid 22 and name NZ*"),
    "R28--E49" : ("resid 49 and name OE*","resid 28 and name NH*"),
    "Q198--E233" : ("resid 198 and name NE2","resid 233 and name OE*"),
    }