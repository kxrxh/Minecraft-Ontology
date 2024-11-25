from wikiparser import WikiParser


parser = WikiParser()
ores_data = parser.get_ores_data()
parser.save_data_to_json(ores_data, "ore_data.json")
