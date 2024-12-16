import pandas as pd
from rdflib import Graph, Namespace
from pathlib import Path


# Load the ontology
def load_ontology():
    g = Graph()
    ontology_path = Path("data/minecraft.owl")
    if ontology_path.exists():
        g.parse(str(ontology_path))
    return g


def create_crafting_dataset(g):
    # Create namespaces to match ontology
    MC = Namespace("http://minecraft.example.org/")
    
    # Query for crafting information with additional properties for ML
    crafting_query = """
    SELECT DISTINCT ?item_name ?item_type ?material_name 
           (SUM(?material_count) as ?total_count)
           ?durability ?attack_damage ?protection_value
    WHERE {
        {
            # Get tools and weapons
            ?item a ?type ;
                  rdfs:label ?item_name ;
                  mc:hasRecipe ?recipe .
            ?recipe mc:usesMaterial ?material .
            ?material rdfs:label ?material_name .
            ?recipe mc:materialCount ?material_count .
            OPTIONAL { ?item mc:hasDurability ?durability }
            OPTIONAL { ?item mc:hasAttackDamage ?attack_damage }
            FILTER(?type IN (mc:Tool, mc:CombatTool))
            # Get most specific type
            FILTER NOT EXISTS {
                ?subtype ^a ?item ;
                         rdfs:subClassOf ?type .
                FILTER (?subtype != ?type)
            }
        }
        UNION
        {
            # Get armor
            ?item a mc:Armor ;
                  rdfs:label ?item_name ;
                  mc:hasRecipe ?recipe .
            ?recipe mc:usesMaterial ?material .
            ?material rdfs:label ?material_name .
            ?recipe mc:materialCount ?material_count .
            OPTIONAL { ?item mc:hasDurability ?durability }
            OPTIONAL { ?item mc:hasProtectionValue ?protection_value }
            BIND(mc:Armor AS ?type)
        }
        
        # Get item type
        BIND(STR(?type) AS ?item_type)
    }
    GROUP BY ?item_name ?item_type ?material_name 
             ?durability ?attack_damage ?protection_value
    ORDER BY ?item_name ?material_name
    """
    
    results = g.query(crafting_query)
    
    # Convert to DataFrame
    df = pd.DataFrame(results, columns=[
        'Item_Name',          # Categorical - for item identification
        'Item_Type',          # Categorical - for item classification
        'Material_Name',      # Categorical - input feature for embedding
        'Material_Count',     # Numerical - input feature
        'Durability',         # Numerical - potential target variable
        'Attack_Damage',      # Numerical - potential target variable
        'Protection_Value'    # Numerical - potential target variable
    ])
    
    # Clean up data types
    df['Material_Count'] = pd.to_numeric(df['Material_Count'], errors='coerce')
    df['Durability'] = pd.to_numeric(df['Durability'], errors='coerce')
    df['Attack_Damage'] = pd.to_numeric(df['Attack_Damage'], errors='coerce')
    df['Protection_Value'] = pd.to_numeric(df['Protection_Value'], errors='coerce')
    
    # Clean up item types
    df['Item_Type'] = df['Item_Type'].str.replace('http://minecraft.example.org/', '')
    
    return df


def main():
    # Load ontology
    g = load_ontology()

    # Create dataset
    crafting_df = create_crafting_dataset(g)

    # Save dataset
    crafting_df.to_csv("data/minecraft_crafting.csv", index=False)

    # Print basic statistics
    print("\nDataset Statistics:")
    print("=" * 50)
    print(f"Dataset shape: {crafting_df.shape}")

    # Display sample information
    print("\nSample Data:")
    print(crafting_df.head())

    # Display value counts for categorical columns
    print("\nItem Types Distribution:")
    print(crafting_df["Item_Type"].value_counts())
    print("\nMaterial Distribution:")
    print(crafting_df["Material_Name"].value_counts())

    # Display basic statistics for numerical columns
    print("\nNumerical Statistics:")
    print(crafting_df.describe())


if __name__ == "__main__":
    main()
