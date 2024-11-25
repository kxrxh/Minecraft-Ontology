from wikiparser import WikiParser

parser = WikiParser()
smelting_data = parser.get_smelting_data()
parser.save_data_to_json(smelting_data, "smelting_data.json")
