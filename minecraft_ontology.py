from rdflib import Graph, Literal, RDF, RDFS, Namespace, URIRef
from rdflib.namespace import XSD
import json
from oreparser import parse_minecraft_ores
from wikiparser import CraftingRecipeParser

# Create namespaces
MC = Namespace("http://minecraft.example.org/")
TOOL = Namespace("http://minecraft.example.org/tool/")
ORE = Namespace("http://minecraft.example.org/ore/")
MATERIAL = Namespace("http://minecraft.example.org/material/")

def create_minecraft_ontology():
    g = Graph()
    
    # Bind namespaces
    g.bind("mc", MC)
    g.bind("tool", TOOL)
    g.bind("ore", ORE)
    g.bind("material", MATERIAL)
    
    # Define main classes
    g.add((MC.Tool, RDF.type, RDFS.Class))
    g.add((MC.Ore, RDF.type, RDFS.Class))
    g.add((MC.Material, RDF.type, RDFS.Class))
    
    # Define properties
    g.add((MC.requiresPickaxe, RDF.type, RDF.Property))
    g.add((MC.hasDurability, RDF.type, RDF.Property))
    g.add((MC.hasVariant, RDF.type, RDF.Property))
    g.add((MC.dropsResource, RDF.type, RDF.Property))
    g.add((MC.hasAbundance, RDF.type, RDF.Property))
    g.add((MC.foundInBiome, RDF.type, RDF.Property))
    g.add((MC.craftedFrom, RDF.type, RDF.Property))
    
    # Load ore data
    ores = parse_minecraft_ores("")
    
    # Add ore information
    for ore_data in ores:
        ore_uri = ORE[ore_data['name'].replace(' ', '_')]
        g.add((ore_uri, RDF.type, MC.Ore))
        g.add((ore_uri, RDFS.label, Literal(ore_data['name'])))
        
        # Add ore properties
        if ore_data['min_pickaxe']:
            g.add((ore_uri, MC.requiresPickaxe, Literal(ore_data['min_pickaxe'])))
        
        for variant in ore_data['variants']:
            g.add((ore_uri, MC.hasVariant, Literal(variant)))
        
        g.add((ore_uri, MC.dropsResource, Literal(ore_data['raw_resource'])))
        g.add((ore_uri, MC.hasAbundance, Literal(ore_data['abundance'])))
        
        for biome in ore_data['biome_restrictions']:
            g.add((ore_uri, MC.foundInBiome, Literal(biome)))
    
    # Load tool data
    parser = CraftingRecipeParser()
    with open('tool_recipes.json', 'r') as f:
        tool_data = json.load(f)
    
    # Add tool information
    for tool_name, tool_info in tool_data.items():
        tool_uri = TOOL[tool_name.replace(' ', '_')]
        g.add((tool_uri, RDF.type, MC.Tool))
        g.add((tool_uri, RDFS.label, Literal(tool_name)))
        
        # Add durability if available
        if 'durability' in tool_info:
            g.add((tool_uri, MC.hasDurability, 
                  Literal(tool_info['durability'], datatype=XSD.integer)))
        
        # Add crafting relationships
        if 'crafting' in tool_info and tool_info['crafting']:
            recipe = tool_info['crafting']
            if 'grid' in recipe:
                for row in recipe['grid']:
                    for item in row:
                        if item and 'item' in item:
                            material_uri = MATERIAL[item['item'].replace(' ', '_')]
                            g.add((tool_uri, MC.craftedFrom, material_uri))
                            g.add((material_uri, RDF.type, MC.Material))
    
    return g

def save_ontology(g, format='xml', filename='minecraft.owl'):
    g.serialize(destination=filename, format=format)

def main():
    # Create and save the ontology
    g = create_minecraft_ontology()
    save_ontology(g)
    
    # Example SPARQL query to find tools that can mine specific ores
    query = """
    SELECT ?tool ?ore
    WHERE {
        ?ore rdf:type mc:Ore ;
             mc:requiresPickaxe ?pickaxe .
        ?tool rdf:type mc:Tool ;
              rdfs:label ?toolName .
        FILTER(CONTAINS(LCASE(?toolName), LCASE(?pickaxe)))
    }
    """
    
    results = g.query(query)
    print("\nTools that can mine ores:")
    for row in results:
        print(f"{row.tool} can mine {row.ore}")

if __name__ == "__main__":
    main() 