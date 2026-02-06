"""
Generate 3D molecule visualizations for LinkedIn/presentations
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

# Read top candidates
df = pd.read_csv('adalia_final_ranked.csv')
top_5 = df.head(5)

print("🎨 Generating 3D molecule images for top 5 candidates...\n")

for idx, row in top_5.iterrows():
    name = row['name']
    smiles = row['smiles']
    
    # Create molecule
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        print(f"❌ Failed: {name}")
        continue
    
    # Add hydrogens and generate 3D coordinates
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.MMFFOptimizeMolecule(mol)
    
    # Save as MOL file (for external rendering)
    mol_filename = f"{name.replace(' ', '_').replace('-', '_')}.mol"
    Chem.MolToMolFile(mol, mol_filename)
    
    # Generate 2D image (PNG for quick view)
    mol_2d = Chem.RemoveHs(mol)
    img = Draw.MolToImage(mol_2d, size=(800, 600))
    img.save(f"{name.replace(' ', '_').replace('-', '_')}_2D.png")
    
    print(f"✅ {name}")
    print(f"   • 3D file: {mol_filename}")
    print(f"   • 2D image: {name.replace(' ', '_').replace('-', '_')}_2D.png\n")

print("🚀 DONE! Use .mol files in ChimeraX or PyMOL for fancy 3D renders like your images.")
print("\nFor LinkedIn: Use the 2D PNG images, or render .mol files in:")
print("  • ChimeraX (free): https://www.cgl.ucsf.edu/chimerax/")
print("  • PyMOL (free): https://pymol.org/")
