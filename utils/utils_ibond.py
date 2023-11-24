from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service

from bs4 import BeautifulSoup
import shutil
from PIL import Image
import time
import random
from pprint import pprint
import pandas as pd
from rdkit import Chem
from pathlib import Path
from IPython.display import SVG
from collections import defaultdict
import re
import numpy as np
import lxml.etree as ET
from rdkit.Chem import PandasTools
from pathlib import Path
import urllib.request
import requests


def split_data(lst_data, lst_split):
    split_data = []
    start = 0
    for length in lst_split:
        split_data.append(lst_data[start : start + length])
        start += length
    return split_data


def get_pka_table_per_page(pka_table_html):
    soup = BeautifulSoup(pka_table_html, "html.parser")
    # header_names = [header_name.get_text(strip=True) for header_name in soup.select('table#pka_table th')]
    # header_names.extend(('ref_link', 'ref_title'))

    # get a list of image links for the pka table
    lst_imgs = [img["src"] for img in soup.select("table#pka_table tbody img")]
    # make a generator for the rowspan values which tells us how many rows each image spans, i.e how many pka values each image has
    gen_rowspan = (
        int(cell["rowspan"])
        for row in soup.select("table#pka_table tbody tr")
        for cell in row.select("td")
        if "rowspan" in cell.attrs
    )

    lst_data = []
    # iterate over each row in the table and extract the data
    for row in soup.select("table#pka_table tbody tr"):
        cells = [cell.get_text(strip=True) for cell in row.select("td")]
        # if the row contains a link, extract the link and the title
        if row.select("a"):
            for attr in row.select("a"):
                cells.extend((attr.attrs["href"], attr.attrs["title"]))
        else:
            # there is sometimes a mistake in the html where the tag is <'adata-toggle="tooltip"'> instead of <'a'>
            # this try/except block is to catch that error and insert empty strings instead if no link or title is found
            try:
                element = driver.find_element(By.TAG_NAME, 'adata-toggle="tooltip"')
                title = element.get_attribute("title")
                link = element.get_attribute("href")
                if link is None:
                    link = ""
                if title is None:
                    title = ""
                cells.extend((link, title))
            except Exception as e:
                print(e)
                print("will insert empty strings")
                cells.extend(("", ""))
        # append the list of cells to the list of data. This will be a list of lists, repressenting the rows of the table
        lst_data.append(cells)

    # removes the first empty string in each list. This is because the first cell in each row is an image, which spans multiple rows
    lst_data = [lst[1:] if lst[0] == "" else lst for lst in lst_data]
    # split the data into a list of lists, where each list represents the data for each image
    lst_split_data = split_data(lst_data=lst_data, lst_split=gen_rowspan)
    # create a dictionary where the keys are the image links and the values are the data for each image
    dict_data = dict(zip(lst_imgs, lst_split_data))

    # Convert the dictionary data into a list of dictionaries using list comprehension
    data_list = [
        {
            "img": img,
            "solvent": row[0],
            "pka": row[1],
            "method": row[2],
            "ref": row[3],
            "ref_link": row[4],
            "ref_title": row[5],
        }
        for img, rows in dict_data.items()
        for row in rows
    ]

    # Create the pandas DataFrame
    df = pd.DataFrame(data_list)

    return df


def get_curent_page_index():
    paginator_html = driver.find_element(By.ID, "paginator")
    current_idx = [
        idx
        for idx, element in enumerate(paginator_html.find_elements(By.TAG_NAME, "a"))
        if "(current)" in element.text
    ][0]
    return current_idx


def get_pka_table(num_pages):
    """_summary_
    Go to a folder on the iBonD website and extract the pKa table data.
    To find out how many pages there are, we need to find the last page number (done manually).

    Args:
        num_pages (_type_): _description_

    Returns:
        _type_: _description_
    """
    # timeout is set to 30 seconds
    wait = WebDriverWait(driver, 30)
    df_pages = []
    for _ in range(num_pages):
        paginator_html = driver.find_element(By.ID, "paginator")
        current_idx, page_number = [
            (idx, page_num)
            for idx, page_num in enumerate(
                paginator_html.find_elements(By.TAG_NAME, "a")
            )
            if "(current)" in page_num.text
        ][0]
        page_number = int(page_number.text.split("\n")[0])
        next_page_idx = (
            current_idx + 2
        )  # current_idx+2 because the first element is the previous page arrow before the page numbers
        # next_page_number = driver.find_element(By.ID, 'paginator').find_element(By.XPATH, f'//ul[@id="paginator"]/li[{current_idx+2}]/a').text
        next_page_number = page_number + 1

        try:
            if page_number == 1:
                print("on page 1")
                page_pka_table_html = driver.find_element(
                    By.ID, "pka_table"
                ).get_attribute("outerHTML")
                df_page = get_pka_table_per_page(pka_table_html=page_pka_table_html)
                df_pages.append((page_number, df_page))
            print(f"going to page: {next_page_number}")
            element = wait.until(
                EC.presence_of_element_located(
                    (By.XPATH, f'//ul[@id="paginator"]/li[{next_page_idx}]/a')
                )
            )
            element.click()
        except Exception as e:
            print(f"cannot find next_page_idx: {next_page_idx}")
            print(driver.find_element(By.ID, "paginator").get_attribute("outerHTML"))
            print(f"next page number is: {next_page_number}")
            print(e)
            break
        time.sleep(10)
        wait.until(EC.presence_of_element_located((By.ID, "pka_table")))
        page_pka_table_html = driver.find_element(By.ID, "pka_table").get_attribute(
            "outerHTML"
        )
        try:
            df_page = get_pka_table_per_page(pka_table_html=page_pka_table_html)
        except Exception as e:
            print("error in function: get_pka_table_per_page")
            print(e)
            break
        df_pages.append((next_page_number, df_page))
        time.sleep(random.randrange(15, 40))

    return df_pages


def download_ibond_image(base_url, img_url, save_path):
    img_name = img_url.split("/")[-1]
    try:
        data = requests.get(base_url + img_url).content
    except Exception as e:
        print(e)
        print(
            f"cannot download image: {img_name}\n url : {base_url+img_url} \n skipping..."
        )
    img_path = str(Path(save_path / img_name))
    print(img_path)
    f = open(img_path, "wb")
    f.write(data)
    f.close()

    return


def download_unique_ibond_images(df, save_path, base_url="http://ibond.nankai.edu.cn"):
    array_unique_imgs_url = df["img"].unique()
    for img_url in array_unique_imgs_url:
        try:
            download_ibond_image(
                base_url=base_url, img_url=img_url, save_path=save_path
            )
        except Exception as e:
            print(f"could not download image: {img_url}")
            print(e)
            continue
        time.sleep(random.randrange(2, 5))
    return


def get_dmso_imgs(df, img_path):
    df_dmso = df.query('Solvent == "DMSO"').copy()
    df_dmso_imgs = df_dmso["img_id"].unique()
    dmso_imgs_path = Path(img_path.parent / "imgs_DMSO")
    Path(dmso_imgs_path).mkdir(parents=True, exist_ok=True)

    for path in Path.glob(img_path, "*.png"):
        img_name_id = path.name.split("_")[0]
        if img_name_id in df_dmso_imgs:
            shutil.copy(path, dmso_imgs_path)
    return


def Reformat_Image(ImageFilePath, OutputFilePath):
    from PIL import Image

    image = Image.open(ImageFilePath, "r")
    image_size = image.size
    width = image_size[0]
    height = image_size[1]

    if width != height:
        bigside = width if width > height else height

        background = Image.new("RGBA", (bigside, bigside), (255, 255, 255, 255))
        offset = (
            int(round(((bigside - width) / 2), 0)),
            int(round(((bigside - height) / 2), 0)),
        )

        background.paste(image, offset)
        background.save(Path(OutputFilePath / "temp.png"), quality=100)
        # print("Image has been resized !")
        return

    else:
        print("Image is already a square, it has not been resized !")
        return


def process_image(path_imgs_raw):
    img_name_processed = (
        str(path_imgs_raw).split("/")[-1].split(".")[0] + "_processed.png"
    )
    path_imgs_processed = Path(path_imgs_raw.parent.parent / "imgs_processed")
    print(path_imgs_processed)
    # Path(path_imgs_processed).mkdir(parents=True, exist_ok=True)
    # image = Image.open(path_imgs_raw)
    Reformat_Image(ImageFilePath=path_imgs_raw, OutputFilePath=path_imgs_processed)
    temp_image = Image.open(Path(path_imgs_processed / "temp.png"))
    new_image = Image.new(
        "RGBA", temp_image.size, "WHITE"
    )  # Create a white rgba background
    new_image.paste(
        temp_image, (0, 0), temp_image
    )  # Paste the image on the background. Go to the links given below for details.
    # remove the temp image
    Path.unlink(Path(path_imgs_processed / "temp.png"), missing_ok=False)
    return new_image.convert("RGB").save(
        f"{path_imgs_processed}/{img_name_processed}", "PNG"
    )  # Save as PNG
