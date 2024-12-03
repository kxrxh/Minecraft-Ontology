from src.parsers.wiki_parser import WikiParser
from src.utils.file_utils import save_json_data
from src.parsers import PROCESSED_DATA_DIR
import time


def run_all_parsers():
    """Run all parsers in sequence and save their data."""
    parser = WikiParser()

    print("\n=== Starting Tool Parser ===")
    tool_data = parser.get_tool_data()
    save_json_data(tool_data, PROCESSED_DATA_DIR, "tool_recipes.json")
    time.sleep(2)  # Add delay between major operations

    print("\n=== Starting Armor Parser ===")
    armor_data = parser.get_armor_data()
    save_json_data(armor_data, PROCESSED_DATA_DIR, "armor_data.json")
    time.sleep(2)

    print("\n=== Starting Ore Parser ===")
    ore_data = parser.get_ores_data()
    save_json_data(ore_data, PROCESSED_DATA_DIR, "ore_data.json")
    time.sleep(2)

    print("\n=== Starting Sword Parser ===")
    sword_data = parser.get_sword_data()
    save_json_data(sword_data, PROCESSED_DATA_DIR, "sword_recipes.json")

    print("\n=== All parsers completed successfully ===")


if __name__ == "__main__":
    run_all_parsers()
