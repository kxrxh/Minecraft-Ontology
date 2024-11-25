from rdflib import OWL, Graph, Literal, RDF, RDFS, Namespace
from rdflib.namespace import XSD
import json
import os
import sys

# Create namespaces
MC = Namespace("http://minecraft.example.org/")
TOOL = Namespace("http://minecraft.example.org/tool/")
ORE = Namespace("http://minecraft.example.org/ore/")
MATERIAL = Namespace("http://minecraft.example.org/material/")
BIOME = Namespace("http://minecraft.example.org/biome/")
LAYER = Namespace("http://minecraft.example.org/layer/")
RECIPE = Namespace("http://minecraft.example.org/recipe/")
ARMOR = Namespace("http://minecraft.example.org/armor/")


def load_json_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: The file '{filepath}' does not exist.", file=sys.stderr)
        sys.exit(1)
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{filepath}': {e}", file=sys.stderr)
        sys.exit(1)


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

    # Define all properties upfront
    properties = {
        "requiresPickaxe": (OWL.ObjectProperty, MC.Ore, MC.Tool),
        "hasDurability": (OWL.DatatypeProperty, MC.Tool, XSD.integer),
        "hasAbundance": (OWL.DatatypeProperty, MC.Ore, XSD.string),
        # ... add all properties here
    }

    for prop_name, (prop_type, domain, range_) in properties.items():
        prop_uri = MC[prop_name]
        g.add((prop_uri, RDF.type, prop_type))
        g.add((prop_uri, RDFS.domain, domain))
        g.add((prop_uri, RDFS.range, range_))

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

    # Load ore data from JSON
    ores_data = load_json_data("ore_data.json")

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

        print("=" * 80)
        g.add((layer_uri, RDF.type, MC.Layer))
        g.add((layer_uri, RDFS.label, Literal(ore_data["Total range"])))
        g.add((ore_uri, MC.foundInLayer, layer_uri))

    # Load tool data from JSON
    tools_data = load_json_data("tool_recipes.json")

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
                material_uri = MATERIAL[material]
                g.add((material_uri, RDF.type, MC.Material))

                # Connect recipe to material
                g.add((recipe_uri, MC.usesMaterial, material_uri))
                g.add(
                    (recipe_uri, MC.materialCount, Literal(count, datatype=XSD.integer))
                )

            # Determine primary material once
            if material_counts:
                primary_material = max(material_counts, key=material_counts.get)
                primary_material_uri = MATERIAL[primary_material]
                g.add((recipe_uri, MC.primaryMaterial, primary_material_uri))

    # Load armor data from JSON
    armor_data = load_json_data("armor_data.json")

    # Add armor information
    for armor_name, armor_info in armor_data.items():
        armor_uri = ARMOR[armor_name.replace(" ", "_")]
        g.add((armor_uri, RDF.type, MC.Armor))
        g.add((armor_uri, RDFS.label, Literal(armor_name)))

        # Add recipe information if available
        if "crafting" in armor_info and "grid" in armor_info["crafting"]:
            recipe_uri = RECIPE[armor_name.replace(" ", "_") + "_Recipe"]
            g.add((recipe_uri, RDF.type, MC.Recipe))
            g.add((armor_uri, MC.hasRecipe, recipe_uri))

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
                material_uri = MATERIAL[material.replace("_", " ")]
                g.add((material_uri, RDF.type, MC.Material))
                g.add((material_uri, RDFS.label, Literal(material.replace("_", " "))))

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
                    primary_material_uri = MATERIAL[primary_material.replace("_", " ")]
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

            # Add protection values
            protection_values = {
                "helmet": {
                    "Leather": 1,
                    "Gold": 2,
                    "Chainmail": 2,
                    "Iron": 2,
                    "Diamond": 3,
                    "Netherite": 3,
                },
                "chestplate": {
                    "Leather": 3,
                    "Gold": 5,
                    "Chainmail": 5,
                    "Iron": 6,
                    "Diamond": 8,
                    "Netherite": 8,
                },
                "leggings": {
                    "Leather": 2,
                    "Gold": 3,
                    "Chainmail": 4,
                    "Iron": 5,
                    "Diamond": 6,
                    "Netherite": 6,
                },
                "boots": {
                    "Leather": 1,
                    "Gold": 1,
                    "Chainmail": 1,
                    "Iron": 2,
                    "Diamond": 3,
                    "Netherite": 3,
                },
            }

            # Add durability values
            durability_values = {
                "Leather": 55,
                "Gold": 77,
                "Chainmail": 166,
                "Iron": 165,
                "Diamond": 363,
                "Netherite": 407,
            }

            # Find slot and add protection/durability
            for slot in armor_slots:
                if slot in armor_name.lower():
                    g.add(
                        (
                            armor_uri,
                            MC.hasProtectionValue,
                            Literal(
                                protection_values[slot][found_material],
                                datatype=XSD.integer,
                            ),
                        )
                    )
                    g.add(
                        (
                            armor_uri,
                            MC.hasBaseDurability,
                            Literal(
                                durability_values[found_material], datatype=XSD.integer
                            ),
                        )
                    )
                    break

    queries = {
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
        "Find ores and the biomes they are found in": """
        SELECT DISTINCT ?ore_name ?biome_name
        WHERE {
            ?ore a mc:Ore ;
                 rdfs:label ?ore_name ;
                 mc:foundInBiome ?biome .
            ?biome rdfs:label ?biome_name .
        }
        ORDER BY ?ore_name ?biome_name
        """,
    }

    return g, queries


def save_ontology(g, format="xml", filename="minecraft.owl"):
    g.serialize(destination=filename, format=format)


def main():
    # Create and save the ontology
    g, queries = create_minecraft_ontology()
    save_ontology(g)

    # Execute and format query results
    for description, query in queries.items():
        print(f"\n{'='*80}")
        print(f"Query: {description}")
        print("=" * 80)

        results = g.query(query)

        # Get column names from the query results
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
            formatted_row = " | ".join(
                f"{str(value):^{widths[col]}}" for col, value in zip(columns, row)
            )
            print(formatted_row)

        print(f"\nTotal results: {len(list(results))}\n")


if __name__ == "__main__":
    main()
