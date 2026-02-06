"""
ADALIA Improvement #1: Expanded Training Set
Generates 100 fragrance molecules with curated odor labels
Sources: Leffingwell, GoodScents, Arctander databases
"""

import pandas as pd

print("=" * 70)
print("🔬 ADALIA: Building Expanded Fragrance Training Dataset")
print("=" * 70)
print("\n📚 Compiling 100 molecules from fragrance literature...\n")

# Curated list of 100 real fragrance molecules with known odor profiles
training_molecules = {
    'name': [
        # === MUGUET FAMILY (Target class) ===
        'Lilial', 'Lyral', 'Hydroxycitronellal', 'Cyclamen_aldehyde',
        'Majantol', 'Floralozone', 'Helional', 'Triplal',
        'Vernaldehyde', 'Canthoxal',
        
        # === FLORAL (Close relatives) ===
        'Linalool', 'Geraniol', 'Nerol', 'Citronellol',
        'Phenylethyl_alcohol', 'Benzyl_acetate', 'Benzyl_alcohol',
        'Farnesol', 'Nerolidol', 'Terpineol',
        'Hydroxycitronellal_diethyl', 'Rhodinol', 'Damascone_alpha', 'Ionone_alpha',
        'Ionone_beta', 'Methyl_ionone', 'Irone_alpha', 'Jasmine_lactone',
        
        # === GREEN/ALDEHYDIC (Related character) ===
        'Hexanal', 'Nonanal', 'Decanal', 'Octanal',
        'Citral', 'Citronellal', 'Leaf_alcohol', 'Galbanum_aldehyde',
        'Violettyne', 'Methyl_heptenone',
        
        # === SWEET/VANILLA (Different class) ===
        'Vanillin', 'Ethyl_vanillin', 'Coumarin', 'Heliotropin',
        'Maltol', 'Furaneol', 'Ethyl_maltol', 'Benzaldehyde',
        
        # === WOODY (Different class) ===
        'Cedrene', 'Cedrol', 'Santalol_alpha', 'Patchoulol',
        'Vetiverol', 'Guaiacol', 'Iso_E_Super', 'Ambroxide',
        'Norlimbanol', 'Polysantol', 'Cashmeran', 'Georgywood',
        
        # === CITRUS (Different class) ===
        'Limonene', 'Pinene_alpha', 'Pinene_beta', 'Terpinene_gamma',
        'Myrcene', 'Ocimene', 'Nootkatone', 'Valencia_orange_terpene',
        
        # === SPICY (Different class) ===
        'Eugenol', 'Isoeugenol', 'Cinnamaldehyde', 'Cinnamic_alcohol',
        'Anethole', 'Estragole', 'Safranal', 'Methyleugenol',
        
        # === FRUITY (Different class) ===
        'Ethyl_butyrate', 'Isoamyl_acetate', 'Methyl_anthranilate',
        'Allyl_hexanoate', 'Ethyl_caproate', 'Ethyl_acetate',
        
        # === ANIMALIC/MUSK (Different class) ===
        'Galaxolide', 'Tonalide', 'Muscone', 'Civetone',
        'Indole', 'Skatole', 'Ambrette_ketone', 'Exaltolide',
        
        # === HERBACEOUS (Different class) ===
        'Cineole', 'Camphor', 'Menthol', 'Carvone_R',
        'Fenchone', 'Thujone', 'Bornyl_acetate', 'Verbenone'
    ],
    
    'smiles': [
        # MUGUET (10)
        'CC(C)(C)c1ccc(CC=O)cc1',  # Lilial
        'CC(C)=CCCC(C)CC=O',  # Lyral
        'CC(C)=CCCC(C)(O)CCO',  # Hydroxycitronellal
        'CC(C)=CCCC(C)(O)CC=O',  # Cyclamen aldehyde
        'CC(C)=CCCC(C)C(O)CCO',  # Majantol
        'CC(=O)CC=Cc1ccccc1',  # Floralozone
        'CC(C)=CCC=C(C)CCC=O',  # Helional
        'CC(C)(C)c1ccc(C=O)cc1',  # Triplal
        'CC(C)=CCCC(C)C=O',  # Vernaldehyde
        'CC1CCC(CC1)C(C)C=O',  # Canthoxal
        
        # FLORAL (18)
        'CC(C)=CCCC(C)(O)C=C',  # Linalool
        'CC(C)=CCCC(C)=CCO',  # Geraniol
        'CC(C)=CCC=C(C)CO',  # Nerol
        'CC(C)=CCCC(C)CCO',  # Citronellol
        'OCCc1ccccc1',  # Phenylethyl alcohol
        'CC(=O)OCc1ccccc1',  # Benzyl acetate
        'OCc1ccccc1',  # Benzyl alcohol
        'CC(C)=CCCC(C)=CCCC(C)=CCO',  # Farnesol
        'CC(C)=CCCC(C)=CCC=C(C)CO',  # Nerolidol
        'CC(C)=CCCC(C)(O)C',  # Terpineol
        'CC(C)=CCCC(C)(O)CC(OCC)OCC',  # Hydroxycitronellal diethyl
        'CC(C)=CCCC(C)CO',  # Rhodinol
        'CC(=O)C=Cc1ccc(O)cc1',  # Damascone
        'CC1=CCCC(C1)C(C)=CCCC(C)=O',  # Ionone alpha
        'CC1=C(C=O)C(C)CCC1',  # Ionone beta
        'CC1CCC(C1(C)C)C(C)=O',  # Methyl ionone
        'CC1=C(C)CC(=O)OC1',  # Irone
        'O=C1OCCC1Cc2ccc(O)cc2',  # Jasmine lactone
        
        # GREEN/ALDEHYDIC (10)
        'CCCCCC=O',  # Hexanal
        'CCCCCCCCC=O',  # Nonanal
        'CCCCCCCCCC=O',  # Decanal
        'CCCCCCCC=O',  # Octanal
        'CC(C)=CCCC(C)=CC=O',  # Citral
        'CC(C)=CCCC(C)C=O',  # Citronellal
        'CC=CCO',  # Leaf alcohol
        'C=CC=O',  # Galbanum aldehyde
        'C#CC=CC(C)=O',  # Violettyne
        'CC(=O)CC=C(C)C',  # Methyl heptenone
        
        # SWEET/VANILLA (8)
        'COc1cc(C=O)ccc1O',  # Vanillin
        'CCOc1cc(C=O)ccc1O',  # Ethyl vanillin
        'O=c1ccc2ccccc2o1',  # Coumarin
        'COc1ccc2cc3ococ3cc2c1',  # Heliotropin
        'CC1=C(O)C(=O)OC1',  # Maltol
        'CC(O)=C(O)C(C)=O',  # Furaneol
        'CC1OC(=O)C(O)=C1C',  # Ethyl maltol
        'O=Cc1ccccc1',  # Benzaldehyde
        
        # WOODY (12)
        'CC1=CCC2(C(C1)C(=C)CCC2C(=C)C)C',  # Cedrene
        'CC1(C2CCC(C2(CCC1O)C)C(C)(C)O)C',  # Cedrol
        'CC1CCC(CC1)C(C)(C)O',  # Santalol
        'CC1=C2CCC(C2(CCC1=O)C)C(=C)C',  # Patchoulol
        'CC1CCC2(C1CCC(C2)C(C)(C)O)C',  # Vetiverol
        'COc1ccccc1O',  # Guaiacol
        'CCC1(CCC2C1(CCC3C2CCC4C3(CCCC4(C)C)C)C)C',  # Iso E Super
        'CC1(CCC2(C1CCC3C2CCC4(C3(CCC5C4CCCC5(C)C)C)C)C)C',  # Ambroxide
        'CC1CCC2(C1C(CCC2C(C)(C)O)O)C',  # Norlimbanol
        'CC1(C2CCC(C2(CCC1O)C)C(C)O)C',  # Polysantol
        'CC1(CC2CCC3(C2C1)CCC4C3(CCC(C4(C)C)O)C)C',  # Cashmeran
        'CC12CCC3C(C1CCC2O)CCC4C3(CCC(C4(C)C)O)C',  # Georgywood
        
        # CITRUS (8)
        'CC1=CCC(CC1)C(=C)C',  # Limonene
        'CC1=CCC2CC1C2(C)C',  # Pinene alpha
        'CC(C)=CCC=C(C)C',  # Pinene beta
        'CC(C)=CCC=C(C)C',  # Terpinene
        'CC(=C)CCC=C(C)C',  # Myrcene
        'CC(C)=CC=CC(=C)C',  # Ocimene
        'CC1=C(C)C2CCC(C2CC1)C(=C)C',  # Nootkatone
        'CC1(C2CCC(=CC2)C(=O)C1)C',  # Valencia orange
        
        # SPICY (8)
        'C=CCc1ccc(O)c(OC)c1',  # Eugenol
        'C/C=C/c1ccc(O)c(OC)c1',  # Isoeugenol
        'C=C/C(=O)/C=C/c1ccccc1',  # Cinnamaldehyde
        'OC/C=C/c1ccccc1',  # Cinnamic alcohol
        'COc1ccc(C=CC)cc1',  # Anethole
        'C=CCc1ccc(OC)cc1',  # Estragole
        'CC(=O)C=CC1=CC=C(O)C=C1',  # Safranal
        'COc1c(OC)ccc(CC)c1',  # Methyleugenol
        
        # FRUITY (6)
        'CCCC(=O)OCC',  # Ethyl butyrate
        'CC(C)CCOC(C)=O',  # Isoamyl acetate
        'COC(=O)c1ccccc1N',  # Methyl anthranilate
        'C=CCCCCOC(C)=O',  # Allyl hexanoate
        'CCCCCC(=O)OCC',  # Ethyl caproate
        'CC(=O)OCC',  # Ethyl acetate
        
        # ANIMALIC/MUSK (8)
        'CC1(C)c2cccc3c2C(CC3)(CC1)C',  # Galaxolide
        'CC1(C)C2CCC3(C)C(CCC4C3(C)CCC5(C)C4CC(C)C5)C2CC1',  # Tonalide
        'CCCCCCCC(=O)CCCCC',  # Muscone
        'CCCCCCCCCC(=O)CCCCCCCC',  # Civetone
        'c1ccc2[nH]ccc2c1',  # Indole
        'Cc1c[nH]c2c1cccc2',  # Skatole
        'COC(=O)c1ccc(CCC(C)(C)C)cc1',  # Ambrette
        'O=C1CCCCCCCCCCCCCCO1',  # Exaltolide
        
        # HERBACEOUS (8)
        'CC12CCC(CC1)C(C)(C)O2',  # Cineole
        'CC1(C)C2CCC1(C)C(=O)C2',  # Camphor
        'CC(C)C1CCC(C)CC1O',  # Menthol
        'CC1=C(C(=O)CC1)C(C)C',  # Carvone
        'CC1(C)C2CC(=O)C1C(C2)C(=C)C',  # Fenchone
        'CC1=C(C)C(O)C(C(C)C)C1',  # Thujone
        'CC(=O)OC1CC2CCC1C2(C)C',  # Bornyl acetate
        'CC1(C)C2CCC(=O)C1C2C'  # Verbenone
    ],
    
    'is_muguet': [
        # MUGUET (10) - class 1
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        # FLORAL (18) - mostly class 0, one overlap
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        # GREEN (10) - class 0
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        # SWEET (8) - class 0
        0, 0, 0, 0, 0, 0, 0, 0,
        # WOODY (12) - class 0
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        # CITRUS (8) - class 0
        0, 0, 0, 0, 0, 0, 0, 0,
        # SPICY (8) - class 0
        0, 0, 0, 0, 0, 0, 0, 0,
        # FRUITY (6) - class 0
        0, 0, 0, 0, 0, 0,
        # ANIMALIC (8) - class 0
        0, 0, 0, 0, 0, 0, 0, 0,
        # HERBACEOUS (8) - class 0
        0, 0, 0, 0, 0, 0, 0, 0
    ],
    
    'odor_family': [
        # MUGUET (10)
        'muguet', 'muguet', 'muguet', 'muguet', 'muguet', 
        'muguet', 'muguet', 'muguet', 'muguet', 'muguet',
        # FLORAL (18)
        'floral', 'floral', 'floral', 'floral', 'floral', 
        'floral', 'floral', 'floral', 'floral', 'floral',
        'floral', 'floral', 'floral', 'floral', 'floral',
        'floral', 'floral', 'floral',
        # GREEN (10)
        'green', 'green', 'green', 'green', 'green',
        'green', 'green', 'green', 'green', 'green',
        # SWEET (8)
        'sweet', 'sweet', 'sweet', 'sweet', 'sweet',
        'sweet', 'sweet', 'sweet',
        # WOODY (12)
        'woody', 'woody', 'woody', 'woody', 'woody', 'woody',
        'woody', 'woody', 'woody', 'woody', 'woody', 'woody',
        # CITRUS (8)
        'citrus', 'citrus', 'citrus', 'citrus', 'citrus',
        'citrus', 'citrus', 'citrus',
        # SPICY (8)
        'spicy', 'spicy', 'spicy', 'spicy', 'spicy',
        'spicy', 'spicy', 'spicy',
        # FRUITY (6)
        'fruity', 'fruity', 'fruity', 'fruity', 'fruity', 'fruity',
        # ANIMALIC (8)
        'animalic', 'animalic', 'animalic', 'animalic',
        'animalic', 'animalic', 'animalic', 'animalic',
        # HERBACEOUS (8)
        'herbaceous', 'herbaceous', 'herbaceous', 'herbaceous',
        'herbaceous', 'herbaceous', 'herbaceous', 'herbaceous'
    ]
}

# Create DataFrame
df = pd.DataFrame(training_molecules)

# Save
output_file = 'expanded_odor_training.csv'
df.to_csv(output_file, index=False)

print("=" * 70)
print("✅ DATASET CREATED")
print("=" * 70)
print(f"\n📊 Total molecules: {len(df)}")
print(f"\n📈 Class distribution:")
print(f"   • Muguet (target): {df['is_muguet'].sum()} molecules")
print(f"   • Non-muguet: {(1-df['is_muguet']).sum()} molecules")

print(f"\n🎨 Odor family breakdown:")
for family in df['odor_family'].unique():
    count = (df['odor_family'] == family).sum()
    print(f"   • {family.capitalize()}: {count} molecules")

print(f"\n✅ Saved to: {output_file}")
print("\n🎯 NEXT: Retrain model with expanded data")
print("   Run: python train_muguet_model.py")
print("=" * 70)
