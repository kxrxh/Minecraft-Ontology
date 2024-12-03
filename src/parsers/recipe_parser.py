from src.parsers.wiki_parser import WikiParser
from src.utils.file_utils import save_json_data
from src.parsers import PROCESSED_DATA_DIR

parser = WikiParser()
recipes = parser.get_tool_data()
save_json_data(recipes, PROCESSED_DATA_DIR, "tool_recipes.json")
