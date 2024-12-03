from rdflib import OWL, Graph, Literal, RDF, RDFS, Namespace
from rdflib.namespace import XSD
import json
import os
import sys
from src.parsers import PROCESSED_DATA_DIR, DATA_DIR
from src.utils.file_utils import load_json_data

# Create namespaces
MC = Namespace("http://minecraft.example.org/")
TOOL = Namespace("http://minecraft.example.org/tool/")
ORE = Namespace("http://minecraft.example.org/ore/")
MATERIAL = Namespace("http://minecraft.example.org/material/")
BIOME = Namespace("http://minecraft.example.org/biome/")
LAYER = Namespace("http://minecraft.example.org/layer/")
RECIPE = Namespace("http://minecraft.example.org/recipe/")
ARMOR = Namespace("http://minecraft.example.org/armor/")


def create_minecraft_ontology():
    g = Graph()

    # Bind namespaces
    g.bind("mc", MC)
    g.bind("tool", TOOL)
    g.bind("ore", ORE)
    g.bind("material", MATERIAL)
    g.bind("biome", BIOME)
    g.bind("layer", LAYER)
    g.bind("recipe", RECIPE)
    g.bind("armor", ARMOR)
    # Define main classes
    g.add((MC.Tool, RDF.type, RDFS.Class))
    g.add((MC.Ore, RDF.type, RDFS.Class))
    g.add((MC.Material, RDF.type, RDFS.Class))
    g.add((MC.Biome, RDF.type, RDFS.Class))
    g.add((MC.Layer, RDF.type, RDFS.Class))
    g.add((MC.Recipe, RDF.type, RDFS.Class))
    g.add((MC.Armor, RDF.type, RDFS.Class))

    # Define subclasses
    g.add((MC.MiningTool, RDFS.subClassOf, MC.Tool))
    g.add((MC.CombatTool, RDFS.subClassOf, MC.Tool))
    g.add((MC.FarmingTool, RDFS.subClassOf, MC.Tool))

    # Add material property relationships
    g.add((MC.canSmelt, RDF.type, OWL.DatatypeProperty))
    g.add((MC.canSmelt, RDFS.domain, MC.Material))
    g.add((MC.canSmelt, RDFS.range, XSD.boolean))

    g.add((MC.smeltsInto, RDF.type, OWL.ObjectProperty))
    g.add((MC.smeltsInto, RDFS.domain, MC.Material))
    g.add((MC.smeltsInto, RDFS.range, MC.Material))

    # Define recipe-related properties
    g.add((MC.usesMaterial, RDF.type, OWL.ObjectProperty))
    g.add((MC.usesMaterial, RDFS.domain, MC.Recipe))
    g.add((MC.usesMaterial, RDFS.range, MC.Material))

    g.add((MC.materialCount, RDF.type, OWL.DatatypeProperty))
    g.add((MC.materialCount, RDFS.domain, MC.Recipe))
    g.add((MC.materialCount, RDFS.range, XSD.integer))

    g.add((MC.primaryMaterial, RDF.type, OWL.ObjectProperty))
    g.add((MC.primaryMaterial, RDFS.domain, MC.Recipe))
    g.add((MC.primaryMaterial, RDFS.range, MC.Material))
    g.add((MC.primaryMaterial, RDFS.subPropertyOf, MC.usesMaterial))

    # Add armor-specific properties
    g.add((MC.hasArmorType, RDF.type, OWL.ObjectProperty))
    g.add((MC.hasArmorType, RDFS.domain, MC.Armor))
    g.add((MC.hasArmorType, RDFS.range, XSD.string))

    g.add((MC.hasArmorSlot, RDF.type, OWL.DatatypeProperty))
    g.add((MC.hasArmorSlot, RDFS.domain, MC.Armor))
    g.add((MC.hasArmorSlot, RDFS.range, XSD.string))

    # Define inverse and functional properties

    # Armor slot relationships (functional - each armor piece goes in exactly one slot)
    g.add((MC.hasArmorSlot, RDF.type, OWL.FunctionalProperty))
    g.add((MC.hasArmorSlot, RDF.type, OWL.ObjectProperty))

    # Recipe relationships (functional - each item has one primary recipe)
    g.add((MC.hasPrimaryRecipe, RDF.type, OWL.FunctionalProperty))
    g.add((MC.hasPrimaryRecipe, RDF.type, OWL.ObjectProperty))
    g.add((MC.hasPrimaryRecipe, RDFS.domain, MC.Armor))
    g.add((MC.hasPrimaryRecipe, RDFS.range, MC.Recipe))

    # Inverse properties for materials and recipes
    g.add((MC.usesMaterial, RDF.type, OWL.ObjectProperty))
    g.add((MC.isUsedInRecipe, RDF.type, OWL.ObjectProperty))
    g.add((MC.usesMaterial, OWL.inverseOf, MC.isUsedInRecipe))

    # Material tier relationships (functional - each material has one tier)
    g.add((MC.hasMaterialTier, RDF.type, OWL.FunctionalProperty))
    g.add((MC.hasMaterialTier, RDF.type, OWL.ObjectProperty))
    g.add((MC.hasMaterialTier, RDFS.domain, MC.Material))
    g.add((MC.hasMaterialTier, RDFS.range, XSD.integer))

    # Armor set relationships
    g.add((MC.isPartOfArmorSet, RDF.type, OWL.ObjectProperty))
    g.add((MC.hasArmorPiece, RDF.type, OWL.ObjectProperty))
    g.add((MC.isPartOfArmorSet, OWL.inverseOf, MC.hasArmorPiece))

    # Protection value (functional - each armor piece has one protection value)
    g.add((MC.hasProtectionValue, RDF.type, OWL.FunctionalProperty))
    g.add((MC.hasProtectionValue, RDF.type, OWL.DatatypeProperty))
    g.add((MC.hasProtectionValue, RDFS.domain, MC.Armor))
    g.add((MC.hasProtectionValue, RDFS.range, XSD.integer))

    # Durability (functional - each armor piece has one base durability)
    g.add((MC.hasBaseDurability, RDF.type, OWL.FunctionalProperty))
    g.add((MC.hasBaseDurability, RDF.type, OWL.DatatypeProperty))
    g.add((MC.hasBaseDurability, RDFS.domain, MC.Armor))
    g.add((MC.hasBaseDurability, RDFS.range, XSD.integer))

    # Add sword-specific properties
    g.add((MC.hasAttackDamage, RDF.type, OWL.DatatypeProperty))
    g.add((MC.hasAttackDamage, RDFS.domain, MC.Tool))
    g.add((MC.hasAttackDamage, RDFS.range, XSD.float))

    g.add((MC.hasAttackSpeed, RDF.type, OWL.DatatypeProperty))
    g.add((MC.hasAttackSpeed, RDFS.domain, MC.Tool))
    g.add((MC.hasAttackSpeed, RDFS.range, XSD.float))

    g.add((MC.hasDPS, RDF.type, OWL.DatatypeProperty))
    g.add((MC.hasDPS, RDFS.domain, MC.Tool))
    g.add((MC.hasDPS, RDFS.range, XSD.float))

    # Load ore data from JSON
    ores_data = load_json_data(PROCESSED_DATA_DIR, "ore_data.json")

    # Add ore information
    for ore_name, ore_data in ores_data.items():
        ore_uri = ORE[ore_name.replace(" ", "_")]
        g.add((ore_uri, RDF.type, MC.Ore))
        g.add((ore_uri, RDFS.label, Literal(ore_name)))

        # Add pickaxe requirement
        if "Minimum pickaxe tier required" in ore_data:
            pickaxe_type = ore_data["Minimum pickaxe tier required"]
            g.add((ore_uri, MC.requiresPickaxe, TOOL[f"{pickaxe_type}_Pickaxe"]))

        # Add abundance
        if "Abundance" in ore_data:
            g.add((ore_uri, MC.hasAbundance, Literal(ore_data["Abundance"])))

        # Add biome information
        if "Found in biome" in ore_data:
            biome_info = ore_data["Found in biome"]
            if biome_info["type"] == "specific":
                for biome in biome_info["biomes"]:
                    biome_uri = BIOME[biome.replace(" ", "_")]
                    g.add((biome_uri, RDF.type, MC.Biome))
                    g.add((ore_uri, MC.foundInBiome, biome_uri))
            else:
                g.add((ore_uri, MC.isUniversal, Literal(True)))

        layer_uri = LAYER[ore_data["Total range"].replace(" ", "_")]
        g.add((layer_uri, RDF.type, MC.Layer))
        g.add((layer_uri, RDFS.label, Literal(ore_data["Total range"])))
        g.add((ore_uri, MC.foundInLayer, layer_uri))

    # Load tool data from JSON
    tools_data = load_json_data(PROCESSED_DATA_DIR, "tool_recipes.json")

    # Add tool and recipe information
    for tool_name, tool_info in tools_data.items():
        tool_uri = TOOL[tool_name.replace(" ", "_")]
        g.add((tool_uri, RDF.type, MC.Tool))
        g.add((tool_uri, RDFS.label, Literal(tool_name)))

        if "durability" in tool_info:
            g.add(
                (
                    tool_uri,
                    MC.hasDurability,
                    Literal(tool_info["durability"], datatype=XSD.integer),
                )
            )

        # Add recipe information
        if "crafting" in tool_info and "grid" in tool_info["crafting"]:
            recipe_uri = RECIPE[tool_name.replace(" ", "_") + "_Recipe"]
            g.add((recipe_uri, RDF.type, MC.Recipe))
            g.add((tool_uri, MC.hasRecipe, recipe_uri))
            g.add((recipe_uri, MC.isRecipeFor, tool_uri))

            # Simplify grid to just item names
            simplified_grid = []
            material_counts = {}

            for row in tool_info["crafting"]["grid"]:
                grid_row = []
                for item in row:
                    if item and "item" in item:
                        material = item["item"]
                        grid_row.append(material)
                        material_counts[material] = material_counts.get(material, 0) + 1
                    else:
                        grid_row.append(None)
                simplified_grid.append(grid_row)

            # Store the simplified grid as a JSON string
            grid_json = json.dumps(simplified_grid)
            g.add((recipe_uri, MC.recipeGrid, Literal(grid_json)))

            # Add material relationships to recipe
            for material, count in material_counts.items():
                material_uri = MATERIAL[material.replace(" ", "_")]
                g.add((material_uri, RDF.type, MC.Material))
                g.add((material_uri, RDFS.label, Literal(material)))

                # Connect recipe to material
                g.add((recipe_uri, MC.usesMaterial, material_uri))
                g.add(
                    (recipe_uri, MC.materialCount, Literal(count, datatype=XSD.integer))
                )

            # Find primary material (most used in recipe)
            if material_counts:
                primary_material = max(material_counts.items(), key=lambda x: x[1])[0]
                primary_material_uri = MATERIAL[primary_material.replace(" ", "_")]
                g.add((recipe_uri, MC.primaryMaterial, primary_material_uri))

    # Load armor data from JSON
    armor_data = load_json_data(PROCESSED_DATA_DIR, "armor_data.json")

    # Add armor information
    for armor_name, armor_info in armor_data.items():
        armor_uri = ARMOR[armor_name.replace(" ", "_")]
        g.add((armor_uri, RDF.type, MC.Armor))
        g.add((armor_uri, RDFS.label, Literal(armor_name)))

        # Add recipe information if available
        if "crafting" in armor_info and "grid" in armor_info["crafting"]:
            recipe_uri = RECIPE[armor_name.replace(" ", "_") + "_Recipe"]
            g.add((recipe_uri, RDF.type, MC.Recipe))
            g.add((armor_uri, MC.hasArmorRecipe, recipe_uri))
            g.add((recipe_uri, MC.isRecipeForArmor, armor_uri))

            # Process crafting grid
            simplified_grid = []
            material_counts = {}

            for row in armor_info["crafting"]["grid"]:
                grid_row = []
                for item in row:
                    if item and "item" in item:
                        material = item["item"]
                        grid_row.append(material)
                        material_counts[material] = material_counts.get(material, 0) + 1
                    else:
                        grid_row.append(None)
                simplified_grid.append(grid_row)

            # Store the simplified grid as a JSON string
            grid_json = json.dumps(simplified_grid)
            g.add((recipe_uri, MC.recipeGrid, Literal(grid_json)))

            # Add material relationships to recipe
            for material, count in material_counts.items():
                material_uri = MATERIAL[material.replace(" ", "_")]
                g.add((material_uri, RDF.type, MC.Material))
                g.add((material_uri, RDFS.label, Literal(material)))

                # Connect recipe to material
                g.add((recipe_uri, MC.usesMaterial, material_uri))
                g.add(
                    (recipe_uri, MC.materialCount, Literal(count, datatype=XSD.integer))
                )

                # Find primary material (most used in recipe)
                if material_counts:
                    primary_material = max(material_counts.items(), key=lambda x: x[1])[
                        0
                    ]
                    primary_material_uri = MATERIAL[primary_material.replace(" ", "_")]
                    g.add((recipe_uri, MC.primaryMaterial, primary_material_uri))

        # Determine material and type
        material_types = {
            "leather": "Leather",
            "golden": "Gold",
            "chainmail": "Chainmail",
            "iron": "Iron",
            "diamond": "Diamond",
            "netherite": "Netherite",
        }

        armor_slots = ["helmet", "chestplate", "leggings", "boots"]

        # Find material and slot
        found_material = None
        for mat_key, mat_name in material_types.items():
            if mat_key in armor_name.lower():
                found_material = mat_name
                material_uri = MATERIAL[mat_name]
                g.add((material_uri, RDF.type, MC.Material))
                g.add((material_uri, RDFS.label, Literal(mat_name)))
                break

        # Add armor set
        if found_material:
            armor_set_uri = ARMOR[f"{found_material}_Set"]
            g.add((armor_set_uri, RDF.type, MC.ArmorSet))
            g.add((armor_set_uri, RDFS.label, Literal(f"{found_material} Set")))
            g.add((armor_uri, MC.isPartOfArmorSet, armor_set_uri))

            # Add material tier
            material_tiers = {
                "Leather": 1,
                "Gold": 2,
                "Chainmail": 3,
                "Iron": 4,
                "Diamond": 5,
                "Netherite": 6,
            }
            g.add(
                (
                    material_uri,
                    MC.hasMaterialTier,
                    Literal(material_tiers[found_material], datatype=XSD.integer),
                )
            )

            # Find slot and add protection/durability
            for slot in armor_slots:
                if slot in armor_name.lower():
                    g.add(
                        (
                            armor_uri,
                            MC.hasProtectionValue,
                            Literal(
                                armor_info["armor_points"],
                                datatype=XSD.integer,
                            ),
                        )
                    )
                    g.add(
                        (
                            armor_uri,
                            MC.hasBaseDurability,
                            Literal(armor_info["durability"], datatype=XSD.integer),
                        )
                    )
                    break

    # Load sword data from JSON
    sword_data = load_json_data(PROCESSED_DATA_DIR, "sword_recipes.json")

    # Add sword information
    for sword_name, sword_info in sword_data.items():
        sword_uri = TOOL[sword_name.replace(" ", "_")]
        g.add((sword_uri, RDF.type, MC.Tool))
        g.add((sword_uri, RDF.type, MC.CombatTool))  # Add sword as combat tool
        g.add((sword_uri, RDFS.label, Literal(sword_name)))

        # Add sword stats
        if "durability" in sword_info:
            g.add(
                (
                    sword_uri,
                    MC.hasDurability,
                    Literal(int(sword_info["durability"]), datatype=XSD.integer),
                )
            )

        if "attack_damage" in sword_info:
            g.add(
                (
                    sword_uri,
                    MC.hasAttackDamage,
                    Literal(float(sword_info["attack_damage"]), datatype=XSD.float),
                )
            )

        if "attack_speed" in sword_info:
            g.add(
                (
                    sword_uri,
                    MC.hasAttackSpeed,
                    Literal(float(sword_info["attack_speed"]), datatype=XSD.float),
                )
            )

        if "dps" in sword_info:
            g.add(
                (
                    sword_uri,
                    MC.hasDPS,
                    Literal(float(sword_info["dps"]), datatype=XSD.float),
                )
            )

        # Add recipe information
        if "crafting" in sword_info and "grid" in sword_info["crafting"]:
            recipe_uri = RECIPE[sword_name.replace(" ", "_") + "_Recipe"]
            g.add((recipe_uri, RDF.type, MC.Recipe))
            g.add((sword_uri, MC.hasRecipe, recipe_uri))
            g.add((recipe_uri, MC.isRecipeFor, sword_uri))

            # Process crafting grid
            simplified_grid = []
            material_counts = {}

            for row in sword_info["crafting"]["grid"]:
                grid_row = []
                for item in row:
                    if item and isinstance(item, dict) and "item" in item:
                        material = item["item"]
                        grid_row.append(material)
                        material_counts[material] = material_counts.get(material, 0) + 1
                    else:
                        grid_row.append(None)
                simplified_grid.append(grid_row)

            # Store the simplified grid as a JSON string
            grid_json = json.dumps(simplified_grid)
            g.add((recipe_uri, MC.recipeGrid, Literal(grid_json)))

            # Add material relationships to recipe
            for material, count in material_counts.items():
                material_uri = MATERIAL[material.replace(" ", "_")]
                g.add((material_uri, RDF.type, MC.Material))
                g.add((material_uri, RDFS.label, Literal(material)))

                # Connect recipe to material
                g.add((recipe_uri, MC.usesMaterial, material_uri))
                g.add(
                    (recipe_uri, MC.materialCount, Literal(count, datatype=XSD.integer))
                )

            # Find primary material (most used in recipe)
            if material_counts:
                primary_material = max(material_counts.items(), key=lambda x: x[1])[0]
                primary_material_uri = MATERIAL[primary_material.replace(" ", "_")]
                g.add((recipe_uri, MC.primaryMaterial, primary_material_uri))

    # Add tool classifications
    farming_tools = ["Hoe", "Shovel"]
    mining_tools = ["Pickaxe", "Axe"]

    # Add material prefixes for tools
    material_prefixes = ["Wooden", "Stone", "Iron", "Golden", "Diamond", "Netherite"]

    # Create and classify all tool variants
    for material in material_prefixes:
        # Add farming tools
        for tool in farming_tools:
            tool_name = f"{material}_{tool}"
            tool_uri = TOOL[tool_name]
            g.add((tool_uri, RDF.type, MC.Tool))
            g.add((tool_uri, RDF.type, MC.FarmingTool))
            g.add((tool_uri, RDFS.label, Literal(f"{material} {tool}")))

        # Add mining tools
        for tool in mining_tools:
            tool_name = f"{material}_{tool}"
            tool_uri = TOOL[tool_name]
            g.add((tool_uri, RDF.type, MC.Tool))
            g.add((tool_uri, RDF.type, MC.MiningTool))
            g.add((tool_uri, RDFS.label, Literal(f"{material} {tool}")))

    return g, {}


def save_ontology(g, format="xml", filename="minecraft.owl"):
    """Save the ontology to the data directory."""
    output_path = os.path.join(DATA_DIR, filename)
    g.serialize(destination=output_path, format=format)
    print(f"Saved ontology to {output_path}")


def get_armor_queries():
    return {
        "Find complete armor sets and their protection values": """
        SELECT DISTINCT ?set_name ?piece_name ?protection ?durability
        WHERE {
            ?armor_set a mc:ArmorSet ;
                      rdfs:label ?set_name .
            ?piece mc:isPartOfArmorSet ?armor_set ;
                   rdfs:label ?piece_name ;
                   mc:hasProtectionValue ?protection ;
                   mc:hasBaseDurability ?durability .
        }
        ORDER BY ?set_name DESC(?protection)
        """,
        "Find armor recipes and their material requirements": """
        SELECT DISTINCT ?armor_name ?material_name ?count
        WHERE {
            ?armor a mc:Armor ;
                   rdfs:label ?armor_name ;
                   mc:hasRecipe ?recipe .
            ?recipe mc:usesMaterial ?material ;
                    mc:materialCount ?count .
            ?material rdfs:label ?material_name .
        }
        ORDER BY ?armor_name ?material_name
        """,
    }


def get_tool_queries():
    return {
        "List all tools with their durability and crafting materials": """
        SELECT DISTINCT ?tool_name ?durability ?material_name ?count
        WHERE {
            ?tool a mc:Tool ;
                  rdfs:label ?tool_name ;
                  mc:hasDurability ?durability ;
                  mc:hasRecipe ?recipe .
            ?recipe mc:usesMaterial ?material ;
                    mc:materialCount ?count .
            ?material rdfs:label ?material_name .
        }
        ORDER BY ?tool_name
        """,
        "List all swords with their combat stats": """
        SELECT DISTINCT ?sword_name ?durability ?attack_damage ?attack_speed ?dps
        WHERE {
            ?sword a mc:CombatTool ;
                   rdfs:label ?sword_name ;
                   mc:hasDurability ?durability .
            OPTIONAL { ?sword mc:hasAttackDamage ?attack_damage }
            OPTIONAL { ?sword mc:hasAttackSpeed ?attack_speed }
            OPTIONAL { ?sword mc:hasDPS ?dps }
            FILTER(CONTAINS(LCASE(?sword_name), "sword"))
        }
        ORDER BY ?sword_name
        """,
    }


def get_mining_queries():
    return {
        "Find Diamond Pickaxe durability": """
        SELECT ?durability
        WHERE {
            ?tool a mc:Tool ;
                  rdfs:label "Diamond Pickaxe" ;
                  mc:hasDurability ?durability .
        }
        """
    }


# Get required tool for every ore
def get_ore_queries():
    return {
        "Find required tool for every ore": """
        SELECT DISTINCT ?ore_name ?tool_name
        WHERE {
            ?ore a mc:Ore ;
                 rdfs:label ?ore_name ;
                 mc:requiresPickaxe ?tool .
            ?tool rdfs:label ?tool_name .
        }
        """,
    }


def can_be_mined_with(g, tool_name, ore_name):
    query = f"""
    SELECT ?required_tier (STR(?tool_name) AS ?tool_tier) (IF(?tool_value >= ?req_value, "Yes", "No") as ?can_mine)
    WHERE {{
        # Get ore and required tool information
        ?ore a mc:Ore ;
             rdfs:label "{ore_name}" ;
             mc:requiresPickaxe ?required_tool .
        ?required_tool rdfs:label ?required_tier .
        
        # Get tool information
        BIND("{tool_name}" AS ?tool_name)
        
        # Define tool tiers mapping
        VALUES (?pickaxe_name ?tool_value) {{
            ("Wooden Pickaxe"    1)
            ("Stone Pickaxe"     2)
            ("Iron Pickaxe"      3)
            ("Golden Pickaxe"    2)  # Golden tools are equivalent to Stone
            ("Diamond Pickaxe"   4)
            ("Netherite Pickaxe" 5)
        }}
        
        # Get required tier value
        VALUES (?req_pickaxe ?req_value) {{
            ("Wooden Pickaxe"    1)
            ("Stone Pickaxe"     2)
            ("Iron Pickaxe"      3)
            ("Diamond Pickaxe"   4)
            ("Netherite Pickaxe" 5)
        }}
        
        # Match the labels
        FILTER(?pickaxe_name = ?tool_name)
        FILTER(?req_pickaxe = ?required_tier)
    }}
    """

    return {f"Can {tool_name} mine {ore_name}?": query}


def get_recipe_for(g, tool_name):
    query = f"""
    SELECT ?recipe ?grid
    WHERE {{
        ?tool rdfs:label "{tool_name}" ;
              mc:hasRecipe ?recipe .
        ?recipe mc:recipeGrid ?grid .
    }}
    """
    return {f"Get recipe for {tool_name}": query}


def execute_query(g, description, query):
    print(f"\n{'='*80}")
    print(f"Query: {description}")
    print("=" * 80)

    results = g.query(query)
    columns = results.vars

    # Calculate column widths
    widths = {col: len(str(col)) for col in columns}
    for row in results:
        for col, value in zip(columns, row):
            widths[col] = max(widths[col], len(str(value)))

    # Print header
    header = " | ".join(f"{col:^{widths[col]}}" for col in columns)
    print("\n" + header)
    print("-" * len(header))

    # Print rows
    for row in results:
        formatted_values = []
        for col, value in zip(columns, row):
            formatted_values.append(f"{str(value):^{widths[col]}}")
        print(" | ".join(formatted_values))

    print(f"\nTotal results: {len(list(results))}\n")


def main():
    # Create and save the ontology
    g, _ = create_minecraft_ontology()
    save_ontology(g)

    print("\nMinecraft Ontology Analysis")
    print("=" * 80)

    # Print ontology statistics
    print(f"\nTotal triples: {len(g)}")
    print(f"Total classes: {len(list(g.subjects(RDF.type, RDFS.Class)))}")
    print(
        f"Total properties: {len(list(g.subjects(RDF.type, OWL.ObjectProperty | OWL.DatatypeProperty)))}"
    )

    # Combine all queries
    all_queries = {}
    all_queries.update(get_armor_queries())
    all_queries.update(get_tool_queries())
    all_queries.update(get_ore_queries())
    all_queries.update(can_be_mined_with(g, "Golden Pickaxe", "Diamond"))
    all_queries.update(can_be_mined_with(g, "Stone Pickaxe", "Diamond"))
    all_queries.update(can_be_mined_with(g, "Iron Pickaxe", "Diamond"))
    all_queries.update(get_recipe_for(g, "Wooden Sword"))

    # Execute all queries
    for description, query in all_queries.items():
        execute_query(g, description, query)


if __name__ == "__main__":
    main()
