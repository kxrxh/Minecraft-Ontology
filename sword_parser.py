from wikiparser import WikiParser

parser = WikiParser()
data = parser.get_sword_data()
parser.save_data_to_json(data, "sword_recipes.json")
