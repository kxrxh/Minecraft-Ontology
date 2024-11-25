import requests
from bs4 import BeautifulSoup
import json
import time
import re


class CraftingRecipeParser:
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
        soup = BeautifulSoup(html_content, 'html.parser')
        tool_links = []
        
        # Define tool keywords to look for
        tool_keywords = [
            'axe', 'hoe', 'pickaxe', 'shovel',
            'fishing rod', 'flint and steel', 'shears',
            'brush', 'spyglass'
        ]
        
        # Find all links in the page
        for link in soup.find_all('a'):
            name = link.get_text(strip=True)
            url = link.get('href')
            
            # Check if the link text contains any tool keywords AND starts with uppercase
            if url and any(keyword in name.lower() for keyword in tool_keywords) and name[0].isupper():
                tool_links.append({
                    'name': name,
                    'url': f"{self.base_url}{url}"
                })
        
        # Remove duplicates while preserving order
        seen = set()
        tool_links = [x for x in tool_links if not (x['name'] in seen or seen.add(x['name']))]
        
        return tool_links

    def parse_recipe(self, tool_page_html):
        """Extracts crafting recipes and durability from a tool's wiki page."""
        soup = BeautifulSoup(tool_page_html, "html.parser")
        data = {
            "crafting": self.parse_crafting_recipe(soup),
            "durability": self.parse_durability(soup)
        }
        return data

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
                                    "item": item_link.get("href", "").replace("/w/", ""),
                                    "image": self.base_url + item_img.get("src", ""),
                                    "title": slot.get("data-minetip-title", ""),
                                    "alt": item_img.get("alt", "")
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
                            "title": (output_slot.find("span", class_="invslot-item") or {}).get("data-minetip-title", ""),
                            "alt": output_img.get("alt", "")
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

    def parse_durability(self, soup):
        """Extracts durability information."""
        durability = None
        # Look for the durability row in the infobox
        durability_row = soup.find(
            "a", string="Durability"
        )
        if durability_row:
            # Get the next td/cell containing the durability value
            durability_cell = durability_row.find_parent("tr")
            if durability_cell:
                # Extract text and parse for Java Edition durability
                text = durability_cell.get_text()
                # Look for pattern like "1561 [JE only]"
                match = re.search(r'(\d+)\s*‌?\[.*JE.*\]', text)
                if match:
                    durability = int(match.group(1))
    
        return durability

    def get_tool_recipes(self):
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
            print(f"Processing {idx}/{len(tool_links)}: {tool['name']}")
            tool_page_html = self.fetch_page(tool["url"])
            if not tool_page_html:
                continue

            data = self.parse_recipe(tool_page_html)
            if data:
                recipes[tool["name"]] = {
                    k: v for k, v in data.items() if v is not None
                }

            time.sleep(1)

        return recipes

    def save_recipes_to_json(self, recipes, filename="tool_recipes.json"):
        """Saves the recipes dictionary to a JSON file."""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(recipes, f, ensure_ascii=False, indent=4)
        print(f"Saved recipes to {filename}")


if __name__ == "__main__":
    parser = CraftingRecipeParser()
    recipes = parser.get_tool_recipes()
    parser.save_recipes_to_json(recipes)
