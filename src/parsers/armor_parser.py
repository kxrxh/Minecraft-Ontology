from src.parsers.wiki_parser import WikiParser
from src.utils.file_utils import save_json_data
from src.parsers import PROCESSED_DATA_DIR

parser = WikiParser()
armor_data = parser.get_armor_data()
save_json_data(armor_data, PROCESSED_DATA_DIR, "armor_data.json")
