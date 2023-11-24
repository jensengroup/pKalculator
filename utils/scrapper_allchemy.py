from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import time
import random
from pprint import pprint
import pandas as pd
from rdkit import Chem
from pathlib import Path

# from IPython.display import SVG
from collections import defaultdict
import re

import lxml.etree as ET


def get_svgs_from_pka_allchemy(smiles, solvent_name="DMSO", random_wait=True):
    """
    smiles: smiles string
    solvent_name: replace with the desired solvent: DMSO, MeCN or THF
    """
    # Set the URL of the website
    url = "http://pka.allchemy.net/"

    # Create a new instance of the Chrome driver (you can use other browsers as well)
    driver = webdriver.Safari()
    # Navigate to the website
    driver.get(url)

    # Find the form elements and interact with them
    smiles_input = driver.find_element_by_id("smiles")
    smiles_input.send_keys(smiles)

    solvent_radio = driver.find_element_by_xpath(
        "//input[@name='solvent' and @value='" + solvent_name + "']"
    )
    solvent_radio.click()

    # Submit the form
    submit_button = driver.find_element_by_xpath("//button[@type='submit']")
    submit_button.click()

    # Wait for the resulting page to load
    wait = WebDriverWait(driver, 10)

    try:
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "svg")))
        svg_elements = driver.find_elements_by_tag_name("svg")
        svgs = [svg_element.get_attribute("outerHTML") for svg_element in svg_elements][
            :2
        ]
    except:
        svgs = "Error getting svgs"

    # Get and print the response content
    # response = driver.page_source
    # print(response)

    if random_wait:
        time.sleep(random.randrange(45, 75))

    # Close the browser
    driver.quit()
    return svgs


def remove_numeric_text_elements(svg, is_string=False):
    # Parse the SVG file
    if is_string:
        root = ET.fromstring(svg, parser=ET.XMLParser(remove_blank_text=True))
    else:
        tree = ET.parse(svg)
        root = tree.getroot()
    removed_count = 0
    # Find and remove <text> elements with numeric values
    for element in root.iter("*"):
        if "ellipse" in element.tag:
            root.remove(element)
        if element.text is not None and element.text.isnumeric():
            parent_text_element = element.getparent()
            root.remove(parent_text_element)
            removed_count += 1

    # Save the modified SVG file
    modified_svg_file_path = str(svg).replace(".svg", "_modified.svg")
    tree.write(modified_svg_file_path)

    print(
        f"Removed {removed_count} <text> elements with numeric values. Modified SVG saved to {modified_svg_file_path}"
    )
    return


def get_svg_mapping(svg, is_string=False):
    """get svg mapping from pka_allchemy svg file
    svg: svg file path or svg string
    is_string: if svg is a string then set to True

    intermediate dictionaries:
        ellipse_color_mapping: key is cx and cy coordinates of the ellipse. value is color code, color name and color description
        tspan_mapping: key is x and y coordinates of the number. value is pka value

    return: defaultdict(list,
            {24: [('#FF0000', 'red', 'most acidic position in a molecule')],
             25: [('#FF0000', 'red', 'most acidic position in a molecule')]})
             where number is pka
    """

    dict_color = {
        "#00FF00": "light green",
        "#009900": "dark green",
        "#33C1EA": "light blue",
        "#FF0000": "red",
        "#FF9999": "pink",
        "#C1C1C1": "grey",
    }

    dict_color_desc = {
        "#00FF00": "deprotonation via chelation controlled process is possible",
        "#009900": "deprotonation via chelation controlled process is possible",
        "#33C1EA": "acidic but unfavourable in lithiation (due to electronic repulsion)",
        "#FF0000": "most acidic position in a molecule",
        "#FF9999": "most acidic position in a molecule",
        "#C1C1C1": "unfavourable due to steric factor",
    }
    # read svg file
    if is_string:
        root = ET.fromstring(svg)
    else:
        tree = ET.parse(svg)
        root = tree.getroot()

    # Define a dictionary to store the number-color mappings
    ellipse_color_mapping = {}

    # Iterate over the ellipse elements
    for ellipse in root.findall(".//{http://www.w3.org/2000/svg}ellipse"):
        cx = float(ellipse.attrib["cx"])
        cy = float(ellipse.attrib["cy"])
        style = ellipse.attrib["style"]

        # Extract the fill color from the style attribute
        fill = style.split(";")[0].split(":")[1]

        # Store the number-color mapping in the dictionary
        ellipse_color_mapping[(cx, cy)] = (
            fill,
            dict_color[fill],
            dict_color_desc[fill],
        )

    tspan_number_mapping = {}
    for text in root.findall(".//{http://www.w3.org/2000/svg}text"):
        if text.getchildren()[0].text.isdigit():
            x = float(text.attrib["x"])
            y = float(text.attrib["y"])
            tspan_number = int(text.getchildren()[0].text)
            tspan_number_mapping[(x, y)] = tspan_number

    tspan_number_ellipse_color_mapping = defaultdict(list)

    for cx, cy in ellipse_color_mapping.keys():
        closest_ellipse = None
        closest_distance = float("inf")
        # get item from number_color_mapping
        for x, y in tspan_number_mapping.keys():
            # print(f'cx {cx} cy {cy}, x {x}, y {y}')
            distance = ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5
            # print(distance)
            if distance < closest_distance:
                closest_ellipse = ellipse_color_mapping[(cx, cy)]
                closest_tspan_number = tspan_number_mapping[(x, y)]
                closest_distance = distance
                # print(f'new closest distance {closest_distance}')
        tspan_number_ellipse_color_mapping[closest_tspan_number].append(closest_ellipse)

    # return ellipse_color_mapping, tspan_number_mapping
    return tspan_number_ellipse_color_mapping


def create_df_from_svgs(folder_path):
    pattern = re.compile(r"[a-z]+([0-9]+)", re.I)
    paths_svg = [
        file for file in folder_path.glob("*.svg") if file.name.split("_")[1] == "0"
    ]
    lst_paths_svg = [
        (match.group(1), path)
        for path in paths_svg
        for match in [pattern.search(path.name)]
        if match
    ]  # .sort(key=lambda x: int(x[0]))
    lst_paths_svg.sort(key=lambda x: int(x[0]))

    dict_mapping = {}
    # itetate through a folder
    for idx, path in lst_paths_svg:
        try:
            svg_mapping = get_svg_mapping(str(path))
            dict_mapping[("_".join(path.name.split("_")[0:2])), idx] = svg_mapping
        except:
            print(f"Error in file: {path}")

    lst_to_df = []
    for key, value in dict_mapping.items():
        lst_to_df.append(
            [key[0].split("_")[0], key[1], min(value.keys()), value[min(value.keys())]]
        )

    df = pd.DataFrame(
        lst_to_df,
        columns=["name", "name_num", "pka_calc", "tspan_number_ellipse_color_mapping"],
    )
    merged_df = df.merge(bordwellCH_rmb_v3_exp, on="name")
    merged_df.sort_index()
    merged_df["pka_error"] = merged_df.apply(
        lambda x: abs(x["pka_exp"] - x["pka_calc"]), axis=1
    )

    return merged_df


if __name__ == "__main__":
    path_data = Path(Path.home() / "pKalculator/data/external/bordwell/")
    path_pka_allchemy = Path(path_data / "pka_allchemy_bordwellCH")
    bordwellCH_rmb_v3_exp = pd.read_csv(Path(path_data / "bordwellCH_rmb_v3.exp"))

    # for comp, smi in bordwellCH_rmb_v3_exp[['name', 'smiles']].values:
    #     svgs = get_svgs_from_pka_allchemy(smiles=smi, random_wait=True)
    #     for idx, svg in enumerate(svgs):
    #         with open(path_pka_allchemy / f'{comp}_{idx}_pka_allchemy.svg', 'w') as f:
    #             f.write(svg)

    df = create_df_from_svgs(path_pka_allchemy)
    print(df)
