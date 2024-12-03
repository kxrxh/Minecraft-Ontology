from src.parsers.wiki_parser import WikiParser
from src.utils.file_utils import save_json_data
from src.parsers import PROCESSED_DATA_DIR

parser = WikiParser()
ores_data = parser.get_ores_data()
save_json_data(ores_data, PROCESSED_DATA_DIR, "ore_data.json")
