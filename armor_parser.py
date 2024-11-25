from wikiparser import WikiParser

parser = WikiParser()
armor_data = parser.get_armor_data()
parser.save_data_to_json(armor_data, "armor_data.json")
