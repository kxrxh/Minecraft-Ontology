from src.parsers.wiki_parser import WikiParser
from src.utils.file_utils import save_json_data
from src.parsers import PROCESSED_DATA_DIR

parser = WikiParser()
data = parser.get_sword_data()
save_json_data(data, PROCESSED_DATA_DIR, "sword_recipes.json")
