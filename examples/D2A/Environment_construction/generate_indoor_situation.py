import random
def generate_bedroom():
    # 必需物品
    bedroom = [
        "A bed with a soft mattress, pillows, and a blanket",
        "A bedside table with a lamp"
    ]
    # 可选物品
    optional_items = [
        "A wardrobe with neatly arranged clothes",
        "A desk with a chair and a reading lamp",
        "A digital alarm clock on the bedside table",
        "A soft area rug under the bed",
        "A potted plant near the window",
        "A bookshelf filled with novels",
        "A cozy armchair with a small side table",
        "A mirror mounted on the wall"
    ]
    # 随机选择一些可选物品
    bedroom += random.sample(optional_items, random.randint(2, 4))
    return bedroom

def generate_kitchen():
    # 必需物品
    kitchen = [
        "A stove for cooking",
        "A fridge stocked with food and beverages",
        "A sink with a tap for washing dishes"
    ]
    # 可选物品
    optional_items = [
        "A microwave oven on the counter",
        "A coffee machine with a variety of coffee pods",
        "A set of pots and pans hanging on a rack",
        "A spice rack with various seasonings",
        "A bread basket with fresh bread",
        "A dining table with chairs",
        "A blender for making smoothies",
        "A pantry filled with canned goods and snacks",
        "A dishwasher under the counter",
        "Herb pots on the windowsill"
    ]
    # 随机选择一些可选物品
    kitchen += random.sample(optional_items, random.randint(3, 5))
    return kitchen

def generate_living_room():
    # 必需物品
    living_room = [
        "A sofa with cushions",
        "A coffee table in front of the sofa"
    ]
    # 可选物品
    optional_items = [
        "A TV on a media console",
        "A bookshelf with decorative items",
        "A rocking chair near the window",
        "A side table with a scented candle",
        "A potted plant in the corner",
        "A floor lamp with adjustable brightness",
        "A rug under the coffee table",
        "A set of board games on a shelf",
        "A wireless speaker for music",
        "A snack bowl filled with chocolates and nuts"
    ]
    # 随机选择一些可选物品
    living_room += random.sample(optional_items, random.randint(3, 5))
    return living_room

def generate_bathroom():
    # 必需物品
    bathroom = [
        "A sink with a mirror above it",
        "A shower with a non-slip mat",
        "A toilet"
    ]
    # 可选物品
    optional_items = [
        "A towel rack with fluffy towels",
        "A cabinet with toiletries",
        "A basket with bath products like bath salts",
        "A small speaker for music",
        "An automatic soap dispenser",
        "A hairdryer stored under the sink",
        "A laundry hamper for used clothes",
        "A decorative plant on the windowsill",
        "A bathrobe hanging on the door",
        "A smart bathroom mirror with LED lighting"
    ]
    # 随机选择一些可选物品
    bathroom += random.sample(optional_items, random.randint(2, 4))
    return bathroom

def generate_house():
    house = {
        "Bedroom": generate_bedroom(),
        "Kitchen": generate_kitchen(),
        "Living Room": generate_living_room(),
        "Bathroom": generate_bathroom()
    }
    return house

def generate_prompt(house):
    prompt = "Generate a detailed description of a house with the following rooms and their respective items. Each room must only include the items specified below. Use vivid and descriptive language to explain the arrangement, functionality, and ambiance of these items.\n\n"

    for room, items in house.items():
        prompt += f"{room}:\n"
        for item in items:
            prompt += f"- {item}\n"
        prompt += "\n"

    return prompt
