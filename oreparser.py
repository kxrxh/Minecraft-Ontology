import json

class OreInfo:
    def __init__(self):
        self.name = ""
        self.variants = []  # stone, deepslate, etc.
        self.raw_resource = ""
        self.min_pickaxe = ""
        self.biome_restrictions = []
        self.abundance = ""
        self.y_levels = ""  # If available
        
    def to_dict(self):
        return {
            "name": self.name,
            "variants": self.variants,
            "raw_resource": self.raw_resource,
            "min_pickaxe": self.min_pickaxe,
            "biome_restrictions": self.biome_restrictions,
            "abundance": self.abundance,
            "y_levels": self.y_levels
        }

def parse_minecraft_ores(wiki_content):
    ores = []
    
    # Known ore types from the wiki
    ore_types = [
        "Coal", "Copper", "Iron", "Gold", "Diamond", 
        "Emerald", "Lapis Lazuli", "Redstone",
        "Nether Gold", "Nether Quartz", "Ancient Debris"
    ]
    
    # Updated biome restrictions mapping
    biome_restrictions = {
        "Emerald": ["Mountains", "Mountain Grove", "Snowy Slopes", "Jagged Peaks", "Frozen Peaks", "Stony Peaks"],
        "Lapis Lazuli": ["Dripstone Caves"],  # Special generation in dripstone caves
        "Nether Gold": ["Nether"],
        "Nether Quartz": ["Nether"],
        "Ancient Debris": ["Nether"]
    }
    
    for ore_name in ore_types:
        ore = OreInfo()
        ore.name = ore_name
        
        # Add basic variants
        if ore_name not in ["Nether Gold", "Nether Quartz", "Ancient Debris"]:
            ore.variants = ["Stone", "Deepslate"]
        elif ore_name == "Ancient Debris":
            ore.variants = ["Netherrack"]
        else:
            ore.variants = ["Netherrack"]
            
        # Set raw resources
        if ore_name in ["Iron", "Gold", "Copper"]:
            ore.raw_resource = f"Raw {ore_name}"
        else:
            ore.raw_resource = ore_name
            
        # Set minimum pickaxe requirements
        if ore_name in ["Coal"]:
            ore.min_pickaxe = "Wooden"
        elif ore_name in ["Copper", "Iron"]:
            ore.min_pickaxe = "Stone"
        elif ore_name in ["Gold", "Diamond", "Redstone", "Emerald", "Lapis Lazuli"]:
            ore.min_pickaxe = "Iron"
        elif ore_name in ["Ancient Debris"]:
            ore.min_pickaxe = "Diamond"
            
        # Set abundance (from wiki data)
        abundances = {
            "Coal": "Very common",
            "Copper": "Common",
            "Iron": "Common",
            "Gold": "Rare",
            "Diamond": "Rare",
            "Emerald": "Very rare",
            "Lapis Lazuli": "Uncommon",
            "Redstone": "Common",
            "Nether Gold": "Common",
            "Nether Quartz": "Common",
            "Ancient Debris": "Very rare"
        }
        ore.abundance = abundances.get(ore_name, "Unknown")
        
        # Updated biome restrictions - now as a list
        ore.biome_restrictions = biome_restrictions.get(ore_name, [])
            
        ores.append(ore.to_dict())
    
    return ores

def save_to_json(ores, filename="minecraft_ores.json"):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({"ores": ores}, f, indent=2)

def main():
    ores = parse_minecraft_ores("")  # Wiki content would go here
    save_to_json(ores)
    print(f"Ore data has been saved to minecraft_ores.json")

if __name__ == "__main__":
    main()
