"""
ADALIA Project: AI-Driven Lilial Replacement Analysis
Computes fragrance-relevant properties for 20 non-toxic muguet analogs
Target: £200k+ licensing to Givaudan/IFF/Firmenich
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Crippen, rdMolDescriptors
from rdkit.Chem import DataStructs
import warnings
warnings.filterwarnings('ignore')  # Suppress RDKit deprecation warnings


# LILIAL REFERENCE (the molecule you're replacing)
LILIAL_SMILES = "CC(C)(C)c1ccc(CC=O)cc1"  # Butylphenyl methylpropional
lilial_mol = Chem.MolFromSmiles(LILIAL_SMILES)
lilial_fp = AllChem.GetMorganFingerprintAsBitVect(lilial_mol, 2, nBits=2048)


def analyze_molecule(smiles):
    """
    ADALIA Property Calculator:
    Compute fragrance-relevant descriptors + Lilial similarity
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return {
            'MW': None, 'LogP': None, 'TPSA': None, 
            'RotBonds': None, 'Lilial_Similarity': None, 
            'Aromatic_Rings': None, 'Has_Aldehyde': None, 
            'Status': 'INVALID'
        }
    
    # Core Properties
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)  # Use Crippen for better accuracy
    tpsa = Descriptors.TPSA(mol)  # Skin penetration proxy
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    
    # Safety Alert: Flag aldehyde (like Lilial's toxic aldehyde)
    aldehyde_smarts = Chem.MolFromSmarts('[CX3H1](=O)[#6]')
    has_aldehyde = 1 if mol.HasSubstructMatch(aldehyde_smarts) else 0
    
    # Lilial Similarity (Tanimoto on Morgan fingerprints)
    candidate_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    similarity = DataStructs.TanimotoSimilarity(lilial_fp, candidate_fp)
    
    # 3D Generation (for packet visualization)
    try:
        mol_3d = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())
        if result == -1:  # Embedding failed
            status = 'NO_3D'
        else:
            AllChem.MMFFOptimizeMolecule(mol_3d)  # Quick geometry optimization
            status = 'OK'
    except:
        status = 'NO_3D'
    
    return {
        'MW': round(mw, 2),
        'LogP': round(logp, 2),
        'TPSA': round(tpsa, 2),
        'RotBonds': rot_bonds,
        'Lilial_Similarity': round(similarity, 3),
        'Aromatic_Rings': aromatic_rings,
        'Has_Aldehyde': has_aldehyde,
        'Status': status
    }


# MAIN EXECUTION
print("=" * 70)
print("🔬 ADALIA PROJECT: Lilial Replacement Analysis")
print("=" * 70)
print(f"📊 Reference: Lilial (MW={round(Descriptors.MolWt(lilial_mol), 2)}, LogP={round(Crippen.MolLogP(lilial_mol), 2)})")
print(f"🎯 Target: Identify top 5 non-toxic analogs for £200k+ licensing\n")

# Load molecules
try:
    df = pd.read_csv('ligands.csv')
    print(f"✅ Loaded {len(df)} candidates from ligands.csv\n")
except FileNotFoundError:
    print("❌ ERROR: ligands.csv not found. Create it first!")
    exit(1)

# Apply analysis
print("🧪 Computing properties...")
results = df['smiles'].apply(analyze_molecule)
result_df = pd.DataFrame(results.tolist())
df = pd.concat([df, result_df], axis=1)

# RANK by composite score
# Weights: 50% odor match, 30% safety (no aldehyde), 20% volatility match
df['Score'] = (
    df['Lilial_Similarity'] * 0.5 +  # Odor match
    (1 - df['Has_Aldehyde']) * 0.3 + # Safety (no aldehyde)
    (1 - abs(df['LogP'] - 3.5) / 5).clip(0, 1) * 0.2  # LogP near Lilial's ~3.5
)
df = df.sort_values('Score', ascending=False)

# Save full results
df.to_csv('adalia_ranked.csv', index=False)
print("✅ Analysis complete!\n")

# Display top 5 for pitches
print("=" * 70)
print("🏆 TOP 5 CANDIDATES FOR £200K PITCH")
print("=" * 70)
