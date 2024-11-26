import requests
from bs4 import BeautifulSoup
import json
import time
import re


class WikiParser:
    def __init__(self, base_url="https://minecraft.wiki"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "CraftingRecipeParser/1.0 (https://example.com)"}
        )

    def fetch_page(self, url):
        """Fetches the content of a webpage."""
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    def get_tool_links(self, html_content):
        """Extract tool links from the wiki page."""
        soup = BeautifulSoup(html_content, "html.parser")
        tool_links = []

        # Define tool keywords to look for
        tool_keywords = [
            "axe",
            "hoe",
            "pickaxe",
            "shovel",
            "fishing rod",
            "flint and steel",
            "shears",
            "brush",
            "spyglass",
        ]

        # Find all links in the page
        for link in soup.find_all("a"):
            name = link.get_text(strip=True)
            url = link.get("href")

            # Check if the link text contains any tool keywords AND starts with uppercase
            if (
                url
                and any(keyword in name.lower() for keyword in tool_keywords)
                and name[0].isupper()
            ):
                tool_links.append({"name": name, "url": f"{self.base_url}{url}"})

        # Remove duplicates while preserving order
        seen = set()
        tool_links = [
            x for x in tool_links if not (x["name"] in seen or seen.add(x["name"]))
        ]

        return tool_links

    def get_armor_links(self, html_content):
        """Extract armor links from the wiki page."""
        soup = BeautifulSoup(html_content, "html.parser")
        armor_links = []

        armor_keywords = [
            "helmet",
            "chestplate",
            "leggings",
            "boots",
            "tunic",
            "cap",
        ]

        for link in soup.find_all("a"):
            name = link.get_text(strip=True)
            url = link.get("href")

            if (
                url
                and any(keyword in name.lower() for keyword in armor_keywords)
                and len(name.split(" ")) == 2
                and name[0].isupper()
            ):
                armor_links.append({"name": name, "url": f"{self.base_url}{url}"})

        seen = set()
        armor_links = [
            x for x in armor_links if not (x["name"] in seen or seen.add(x["name"]))
        ]

        return armor_links

    def parse_recipe(self, tool_page_html):
        """Extracts crafting recipes, durability, and armor points from a tool/armor wiki page."""
        soup = BeautifulSoup(tool_page_html, "html.parser")
        data = {
            "crafting": self.parse_crafting_recipe(soup),
            "durability": self.parse_durability(soup),
            "armor_points": self.parse_armor_points(soup),
        }
        return {k: v for k, v in data.items() if v is not None}

    def parse_crafting_recipe(self, soup):
        """Extracts crafting recipe information including item images and links."""
        recipe = {}
        crafting_header = soup.find(
            ["span", "h2", "h3"], string=lambda text: text and "Crafting" in text
        )
        if not crafting_header:
            return None

        # Find the crafting grid (mcui-input span)
        crafting_grid = soup.find("span", class_="mcui-input")
        if crafting_grid:
            grid = []
            # Process each row in the crafting grid
            for row in crafting_grid.find_all("span", class_="mcui-row"):
                grid_row = []
                # Process each slot in the row
                for slot in row.find_all("span", class_="invslot"):
                    item_data = {}
                    # Find item image if it exists
                    item_img = slot.find("img")
                    if item_img:
                        item_link = slot.find("a")
                        # Only process if both image and link exist
                        if item_link and item_img:
                            try:
                                item_data = {
                                    "item": item_link.get("href", "").replace(
                                        "/w/", ""
                                    ),
                                    "image": self.base_url + item_img.get("src", ""),
                                    "title": slot.get("data-minetip-title", ""),
                                    "alt": item_img.get("alt", ""),
                                }
                            except AttributeError:
                                # If any attribute access fails, create empty item_data
                                item_data = {}
                    grid_row.append(item_data if item_data else None)
                grid.append(grid_row)

            # Find the output item
            output_slot = soup.find("span", class_="mcui-output")
            if output_slot:
                output_img = output_slot.find("img")
                output_link = output_slot.find("a")
                if output_img and output_link:
                    try:
                        recipe["output"] = {
                            "item": output_link.get("href", "").replace("/w/", ""),
                            "image": self.base_url + output_img.get("src", ""),
                            "title": (
                                output_slot.find("span", class_="invslot-item") or {}
                            ).get("data-minetip-title", ""),
                            "alt": output_img.get("alt", ""),
                        }
                    except AttributeError:
                        recipe["output"] = None

            recipe["grid"] = grid
            recipe["type"] = "crafting"
            return recipe

        # Fallback to old table parsing if no crafting grid is found
        crafting_table = crafting_header.find_next("table", {"class": "wikitable"})
        if not crafting_table:
            return None

        grid = []
        for row in crafting_table.find_all("tr"):
            cols = row.find_all(["td", "th"])
            row_data = [col.get_text(strip=True) for col in cols]
            if any(row_data):
                grid.append(row_data)

        recipe["grid"] = grid
        recipe["type"] = "crafting"
        return recipe

    def parse_durability(self, soup) -> str | None:
        """Extracts durability information from the infobox."""
        durability = None
        infobox = soup.find("table", class_="infobox-rows")
        if infobox:
            for row in infobox.find_all("tr"):
                if row.find("a", string="Durability"):
                    durability = row.find("td").find("p").get_text(strip=True)
                    # Extract just the number from the beginning
                    durability = ''.join(filter(str.isdigit, durability.split()[0]))
                    break
        return int(durability) if durability else None

    def parse_armor_points(self, soup):
        """Extracts armor points from the infobox."""
        armor_points = None
        infobox = soup.find("table", class_="infobox-rows")
        if infobox:
            for row in infobox.find_all("tr"):
                if row.find("a", string="Armor"):
                    armor_points = row.find("td").find("p").get_text(strip=True)
                    if armor_points:
                        armor_points = ''.join(filter(str.isdigit, armor_points.split()[0]))
                        break
        return armor_points

    def clean_name(self, name):
        """Cleans special characters from names."""
        # Remove special characters and normalize spaces
        cleaned = re.sub(r'[\'"`]', "", name)  # Remove quotes
        cleaned = re.sub(
            r"[^a-zA-Z0-9\s-]", "", cleaned
        )  # Keep only alphanumeric, spaces, and hyphens
        cleaned = re.sub(r"\s+", " ", cleaned).strip()  # Normalize spaces
        return cleaned

    def get_tool_data(self):
        """Main method to get all tool data."""
        item_page_url = f"{self.base_url}/w/Item"
        print(f"Fetching Item page: {item_page_url}")
        item_page_html = self.fetch_page(item_page_url)
        if not item_page_html:
            return {}

        tool_links = self.get_tool_links(item_page_html)
        print(f"Found {len(tool_links)} tools.")

        recipes = {}
        for idx, tool in enumerate(tool_links, 1):
            # Clean the tool name
            clean_tool_name = self.clean_name(tool["name"])
            print(f"Processing {idx}/{len(tool_links)}: {clean_tool_name}")

            tool_page_html = self.fetch_page(tool["url"])
            if not tool_page_html:
                continue

            data = self.parse_recipe(tool_page_html)
            if data:
                recipes[clean_tool_name] = {
                    k: v for k, v in data.items() if v is not None
                }

            time.sleep(1)

        return recipes

    def get_armor_data(self):
        """Gets armor data from the wiki."""
        armor_url = f"{self.base_url}/w/Armor"
        print(f"Fetching Armor page: {armor_url}")
        armor_page_html = self.fetch_page(armor_url)
        if not armor_page_html:
            return {}

        armor_links = self.get_armor_links(armor_page_html)
        print(f"Found {len(armor_links)} armor.")

        recipes = {}
        for idx, armor in enumerate(armor_links, 1):
            # Clean the tool name
            clean_armor_name = self.clean_name(armor["name"])
            print(f"Processing {idx}/{len(armor_links)}: {clean_armor_name}")

            armor_page_html = self.fetch_page(armor["url"])
            if not armor_page_html:
                continue

            data = self.parse_recipe(armor_page_html)
            if data:
                recipes[clean_armor_name] = {
                    k: v for k, v in data.items() if v is not None
                }

            time.sleep(1)

        return recipes

    def save_data_to_json(
        self, recipes: dict, filename: str = "tool_recipes.json"
    ) -> None:
        """Saves the recipes dictionary to a JSON file.

        Args:
            recipes: Dictionary containing recipe data to save
            filename: Name of the JSON file to save to. Defaults to "tool_recipes.json"
        """
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(recipes, f, ensure_ascii=False, indent=4)
        print(f"Saved recipes to {filename}")

    def parse_ore_properties(self, ore_page_html):
        """Extracts ore properties like hardness, blast resistance, tool requirements."""
        soup = BeautifulSoup(ore_page_html, "html.parser")
        properties = {
            "hardness": None,
            "blast_resistance": None,
            "tool_required": None,
            "renewable": False,
        }

        # Find the properties table
        info_table = soup.find("table", class_="infobox-rows")
        if info_table:
            rows = info_table.find_all("tr")
            for row in rows:
                header = row.find("th")
                value = row.find("td")
                if header and value:
                    header_text = header.get_text(strip=True).lower()
                    value_text = value.get_text(strip=True)

                    if "hardness" in header_text:
                        properties["hardness"] = value_text
                    elif "blast resistance" in header_text:
                        properties["blast_resistance"] = value_text
                    elif "tool" in header_text:
                        properties["tool_required"] = value_text
                    elif "renewable" in header_text:
                        properties["renewable"] = value_text.lower() == "yes"

        return properties

    def parse_ore_drops(self, ore_page_html):
        """Extracts information about what items the ore drops."""
        soup = BeautifulSoup(ore_page_html, "html.parser")
        drops = {"normal": [], "silk_touch": [], "fortune": {}}

        # Look for drop information in paragraphs and lists
        drop_section = soup.find(
            ["div", "section"], string=lambda text: text and "drops" in text.lower()
        )
        if drop_section:
            # Parse normal drops
            drop_text = drop_section.get_text()
            if "drops" in drop_text.lower():
                # Extract drop information using regex patterns
                normal_drops = re.findall(
                    r"drops (\d+(?:-\d+)?) (.+?)(?=\.|$)", drop_text
                )
                for amount, item in normal_drops:
                    drops["normal"].append({"item": item.strip(), "amount": amount})

        return drops

    def clean_text(self, text):
        """Helper function to clean text by removing footnote markers and extra whitespace"""
        # Remove footnote markers like [a], [JE], [c], [f], etc.
        cleaned = re.sub(r"\[\w+\]", "", text)
        # Remove extra whitespace and strip
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def parse_ore_table(self, html_content):
        """Parses the main ore comparison table from the Minecraft Wiki Ores page."""
        soup = BeautifulSoup(html_content, "html.parser")
        ores_data = {}

        # Find the specific table
        table = soup.find(
            "table", attrs={"data-description": "Ores, resources and mineral blocks"}
        )
        if not table:
            return ores_data

        # Get all rows
        rows = table.find_all("tr")
        if not rows:
            return ores_data

        # Process header row to get ore names and column spans
        header_row = rows[0]
        column_info = []
        current_col = 0

        for cell in header_row.find_all(["th", "td"]):
            if cell == header_row.find_all(["th", "td"])[0]:  # Skip first empty cell
                continue

            colspan = int(cell.get("colspan", 1))
            ore_name = self.clean_text(cell.get_text(strip=True))

            # Add column info for each column (accounting for colspan)
            for _ in range(colspan):
                column_info.append({"ore_name": ore_name, "original_col": current_col})
                current_col += 1

        # Initialize data structure for each ore
        for info in column_info:
            if info["ore_name"] not in ores_data:
                ores_data[info["ore_name"]] = {}

        # Process each row
        for row in rows[1:]:  # Skip header row
            cells = row.find_all(["th", "td"])
            if not cells:
                continue

            # Get the category from the first cell
            category = self.clean_text(cells[0].get_text(strip=True))

            # Process each cell in the row
            current_col = 0
            for cell in cells[1:]:  # Skip first cell (category)
                colspan = int(cell.get("colspan", 1))

                # Get the ore names this cell applies to
                affected_ores = set()
                for _ in range(colspan):
                    if current_col < len(column_info):
                        affected_ores.add(column_info[current_col]["ore_name"])
                    current_col += 1

                # Process cell value based on category
                if category == "Found in biome":
                    biome_data = self.process_biomes(cell)
                    for ore in affected_ores:
                        ores_data[ore][category] = biome_data
                elif category == "Raw resource":
                    resource_data = self.process_raw_resource(cell)
                    for ore in affected_ores:
                        ores_data[ore][category] = resource_data
                elif category == "Minimum pickaxe tier required":
                    pickaxe_tier = self.clean_text(cell.get_text(strip=True))
                    for ore in affected_ores:
                        ores_data[ore][category] = pickaxe_tier
                elif category in [
                    "Total range",
                    "Most found in layers",
                    "None at layers",
                ]:
                    # Handle ranges and layer information
                    value = self.clean_text(cell.get_text(strip=True))
                    for ore in affected_ores:
                        ores_data[ore][category] = value
                elif category == "Abundance":
                    abundance = self.clean_text(cell.get_text(strip=True))
                    for ore in affected_ores:
                        ores_data[ore][category] = abundance

        return ores_data

    def process_biomes(self, cell):
        """Helper function to process biome cell into structured data"""
        biomes = []
        sprite_text_spans = cell.find_all("span", class_="sprite-text")

        if sprite_text_spans:
            # Process each sprite-text span which contains the biome name
            for span in sprite_text_spans:
                biome_name = self.clean_text(span.get_text(strip=True))
                if biome_name and biome_name.lower() != "any":
                    biomes.append(biome_name)

        # If no valid biomes found and text is "any", return universal type
        if not biomes:
            text = self.clean_text(cell.get_text(strip=True)).lower()
            if text == "any":
                return {"type": "universal"}
            # If there's other text, add it as a biome
            if text:
                biomes.append(text)

        return (
            {"type": "specific", "biomes": biomes} if biomes else {"type": "universal"}
        )

    def process_raw_resource(self, cell):
        """Helper function to extract raw resource name from link"""
        link = cell.find("a")
        if link:
            # Get the href and remove '/w/' prefix
            href = link.get("href", "").replace("/w/", "")
            if href:
                return href
        # Fallback to cleaned text if no link found
        return self.clean_text(cell.get_text(strip=True))

    def get_ores_data(self):
        """Main method to get all ore data from the wiki."""
        ores_url = f"{self.base_url}/w/Ore"
        print(f"Fetching Ores page: {ores_url}")
        page_html = self.fetch_page(ores_url)
        if not page_html:
            return {}

        ores_data = self.parse_ore_table(page_html)
        return ores_data

    def get_sword_links(self):
        """Return hardcoded sword links."""
        return [
            {"name": "Wooden Sword", "url": f"{self.base_url}/w/Wooden_Sword"},
            {"name": "Stone Sword", "url": f"{self.base_url}/w/Stone_Sword"},
            {"name": "Iron Sword", "url": f"{self.base_url}/w/Iron_Sword"},
            {"name": "Golden Sword", "url": f"{self.base_url}/w/Golden_Sword"},
            {"name": "Diamond Sword", "url": f"{self.base_url}/w/Diamond_Sword"},
            {"name": "Netherite Sword", "url": f"{self.base_url}/w/Netherite_Sword"},
        ]

    def parse_sword_stats_from_file(self, html_file="sword_damage.html"):
        """Extract sword statistics from the sword_damage.html file."""
        try:
            with open(html_file, "r", encoding="utf-8") as f:
                html_content = f.read()
        except FileNotFoundError:
            print(f"Error: {html_file} not found")
            return {}

        soup = BeautifulSoup(html_content, "html.parser")
        sword_stats = {}

        # Find the sword stats table
        table = soup.find(
            "table", attrs={"data-description": "Sword attack damage by type"}
        )
        if not table:
            return sword_stats

        # Get all rows
        rows = table.find_all("tr")
        if len(rows) < 2:
            return sword_stats

        # Get material names from header row
        headers = rows[0].find_all("th")
        materials = []
        for header in headers[1:]:  # Skip first header (Material)
            material_text = header.find("span", class_="sprite-text")
            if material_text:
                materials.append(material_text.get_text(strip=True))

        # Define the stats we want to extract
        stat_rows = {
            "Attack Damage": "attack_damage",
            "Attack Speed": "attack_speed",
            "Damage/Second (DPS)": "dps",
            "Durability": "durability",
        }

        # Process each stat row
        for row in rows[1:]:  # Skip header row
            cells = row.find_all(["th", "td"])
            if len(cells) < 2:
                continue

            stat_name = cells[0].get_text(strip=True)
            if stat_name not in stat_rows:
                continue

            # Process each material's value for this stat
            for material, cell in zip(materials, cells[1:]):
                if material not in sword_stats:
                    sword_stats[material] = {}

                # Clean and convert the value
                value = (
                    cell.get_text(strip=True).split("×")[0].strip()
                )  # Remove "× N hearts" suffix
                try:
                    value = float(value)
                    if value.is_integer():
                        value = int(value)
                except ValueError:
                    pass  # Keep as string if conversion fails

                sword_stats[material][stat_rows[stat_name]] = value

        return sword_stats

    def get_sword_data(self):
        """Main method to get all sword data."""
        # First get the basic sword data (recipes etc.)
        item_page_url = f"{self.base_url}/w/Item"
        print(f"Fetching Item page: {item_page_url}")
        item_page_html = self.fetch_page(item_page_url)
        if not item_page_html:
            return {}

        sword_links = self.get_sword_links()
        print(f"Found {len(sword_links)} swords.")

        # Get sword stats from the local HTML file
        sword_stats = self.parse_sword_stats_from_file()

        recipes = {}
        for idx, sword in enumerate(sword_links, 1):
            clean_sword_name = self.clean_name(sword["name"])
            print(f"Processing {idx}/{len(sword_links)}: {clean_sword_name}")

            sword_page_html = self.fetch_page(sword["url"])
            if not sword_page_html:
                continue

            # Get crafting recipe
            data = self.parse_recipe(sword_page_html)

            # Add sword stats from the local file
            material = clean_sword_name.split()[
                0
            ]  # Get material name (e.g., "Wooden" from "Wooden Sword")
            if material in sword_stats:
                data.update(sword_stats[material])

            if data:
                recipes[clean_sword_name] = {
                    k: v for k, v in data.items() if v is not None
                }

            time.sleep(1)

        return recipes
