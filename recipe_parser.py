from wikiparser import WikiParser


parser = WikiParser()
recipes = parser.get_tool_data()
parser.save_data_to_json(recipes)
