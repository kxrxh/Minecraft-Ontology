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


def get_recipe_for(tool_name):
    query = f"""
    SELECT ?recipe ?grid
    WHERE {{
        ?tool rdfs:label "{tool_name}" ;
              mc:hasRecipe ?recipe .
        ?recipe mc:recipeGrid ?grid .
    }}
    """
    return {f"Get recipe for {tool_name}": query}


def get_mining_capability_query(tool_name="Diamond Pickaxe", ore_name="Iron Ore"):
    # Query 1: Get required tool tier for the ore
    # TODO: This query returns 0. Fix it!!
    ore_query = f"""
    SELECT ?required_tool
    WHERE {{
        ?ore a mc:Ore ;
             rdfs:label "{ore_name}" ;
             mc:requiresPickaxe ?required_tool .
    }}
    """

    # Query 2: Get tool value mapping
    tool_value_query = """
    SELECT ?pickaxe_name ?tool_value 
    WHERE {
        VALUES (?pickaxe_name ?tool_value) {
            ("Wooden Pickaxe"    1)
            ("Stone Pickaxe"     2)
            ("Iron Pickaxe"      3)
            ("Golden Pickaxe"    2)
            ("Diamond Pickaxe"   4)
            ("Netherite Pickaxe" 5)
        }
    }
    """

    # Query 3: Get required tier value mapping
    req_value_query = """
    SELECT ?req_pickaxe ?req_value
    WHERE {
        VALUES (?req_pickaxe ?req_value) {
            ("Wooden Pickaxe"    1)
            ("Stone Pickaxe"     2)
            ("Iron Pickaxe"      3)
            ("Diamond Pickaxe"   4)
            ("Netherite Pickaxe" 5)
        }
    }
    """

    # Combined final query
    query = f"""
    SELECT ?required_tier (STR(?tool_name) AS ?tool_tier) (IF(?tool_value >= ?req_value, "Yes", "No") as ?can_mine)
    WHERE {{
        ?ore a mc:Ore ;
             rdfs:label "{ore_name}" ;
             mc:requiresPickaxe ?required_tool .
        ?required_tool rdfs:label ?required_tier .
        
        BIND("{tool_name}" AS ?tool_name)
        
        VALUES (?pickaxe_name ?tool_value) {{
            ("Wooden Pickaxe"    1)
            ("Stone Pickaxe"     2)
            ("Iron Pickaxe"      3)
            ("Golden Pickaxe"    2)
            ("Diamond Pickaxe"   4)
            ("Netherite Pickaxe" 5)
        }}
        
        VALUES (?req_pickaxe ?req_value) {{
            ("Wooden Pickaxe"    1)
            ("Stone Pickaxe"     2)
            ("Iron Pickaxe"      3)
            ("Diamond Pickaxe"   4)
            ("Netherite Pickaxe" 5)
        }}
        
        FILTER(?pickaxe_name = ?tool_name)
        FILTER(?req_pickaxe = ?required_tier)
    }}
    """
    return {
        f"Debug - Required tier for {ore_name}": ore_query,
        f"Debug - Tool value mapping": tool_value_query, 
        f"Debug - Required tier mapping": req_value_query,
        f"Can {tool_name} mine {ore_name}?": query
    }


def get_ore_mining_tools_query(ore_name="Diamond Ore"):
    return {
        f"2. What tools can mine {ore_name}?": f"""
        SELECT DISTINCT ?tool_name
        WHERE {{
            ?ore a mc:Ore ;
                 rdfs:label "{ore_name}" ;
                 mc:requiresPickaxe ?required_tool .
            ?required_tool rdfs:label ?required_tier .
            
            VALUES (?pickaxe_name ?tool_value) {{
                ("Wooden Pickaxe"    1)
                ("Stone Pickaxe"     2)
                ("Iron Pickaxe"      3)
                ("Golden Pickaxe"    2)
                ("Diamond Pickaxe"   4)
                ("Netherite Pickaxe" 5)
            }}
            
            VALUES (?req_pickaxe ?req_value) {{
                ("Wooden Pickaxe"    1)
                ("Stone Pickaxe"     2)
                ("Iron Pickaxe"      3)
                ("Diamond Pickaxe"   4)
                ("Netherite Pickaxe" 5)
            }}
            
            FILTER(?req_pickaxe = ?required_tier)
            FILTER(?tool_value >= ?req_value)
            BIND(?pickaxe_name AS ?tool_name)
        }}
        ORDER BY ?tool_name
        """
    }


def get_item_recipe_query(item_name="Diamond Sword"):
    return {
        f"3. What is the recipe for {item_name}?": f"""
        SELECT ?material_name ?count ?grid
        WHERE {{
            ?item rdfs:label "{item_name}" ;
                  mc:hasRecipe ?recipe .
            ?recipe mc:usesMaterial ?material ;
                    mc:materialCount ?count ;
                    mc:recipeGrid ?grid .
            ?material rdfs:label ?material_name .
        }}
        ORDER BY ?material_name
        """
    }


def get_item_durability_query(item_name="Diamond Pickaxe"):
    return {
        f"4. What is the durability of {item_name}?": f"""
        SELECT ?durability
        WHERE {{
            ?item rdfs:label "{item_name}" ;
                  mc:hasDurability ?durability .
        }}
        """
    }


def get_sword_damage_query(sword_name="Diamond Sword"):
    return {
        f"5. What is the attack damage of {sword_name}?": f"""
        SELECT ?attack_damage
        WHERE {{
            ?sword a mc:CombatTool ;
                   rdfs:label "{sword_name}" ;
                   mc:hasAttackDamage ?attack_damage .
        }}
        """
    }


def get_crafting_requirement_query(item_name="Diamond Sword", material_name="Stick"):
    return {
        f"6. Is {material_name} required for crafting {item_name}?": f"""
        SELECT (COUNT(?recipe) > 0 as ?is_required)
        WHERE {{
            ?item rdfs:label "{item_name}" ;
                  mc:hasRecipe ?recipe .
            ?recipe mc:usesMaterial ?required_item .
            ?required_item rdfs:label "{material_name}" .
        }}
        """
    }


def get_crafting_yield_query(item_name="Diamond Sword", material_name="Diamond"):
    return {
        f"7. How many {item_name}s can be crafted from {material_name}?": f"""
        SELECT (1/?count as ?items_per_material)
        WHERE {{
            ?item rdfs:label "{item_name}" ;
                  mc:hasRecipe ?recipe .
            ?recipe mc:usesMaterial ?material ;
                    mc:materialCount ?count .
            ?material rdfs:label "{material_name}" .
        }}
        """
    }


def get_armor_set_pieces_query(set_name="Diamond Set"):
    return {
        f"8. What items are part of {set_name}?": f"""
        SELECT ?piece_name
        WHERE {{
            ?armor_set a mc:ArmorSet ;
                      rdfs:label "{set_name}" .
            ?piece mc:isPartOfArmorSet ?armor_set ;
                   rdfs:label ?piece_name .
        }}
        ORDER BY ?piece_name
        """
    }


def get_sword_dps_query(sword_name="Diamond Sword"):
    return {
        f"9. What is the DPS of {sword_name}?": f"""
        SELECT ?dps
        WHERE {{
            ?sword a mc:CombatTool ;
                   rdfs:label "{sword_name}" ;
                   mc:hasDPS ?dps .
        }}
        """
    }


def get_competence_queries():
    queries = {}
    queries.update(get_mining_capability_query())
    queries.update(get_ore_mining_tools_query())
    queries.update(get_item_recipe_query())
    queries.update(get_item_durability_query())
    queries.update(get_sword_damage_query())
    queries.update(get_crafting_requirement_query())
    queries.update(get_crafting_yield_query())
    queries.update(get_armor_set_pieces_query())
    queries.update(get_sword_dps_query())
    return queries