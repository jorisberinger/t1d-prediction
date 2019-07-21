import logging
import json
import re


class Profile_reader:
    insulin_sensitivity: float
    carb_ratio: float
    id: str

    # convert filename and read profile json to set insulin sensitivity and carb ratio
    def __init__(self, filename:str):
        logging.info("Init profile reader with filename {}".format(filename))
        # convert filename from csv to profile json
        json_filename = self.convert_filename(filename)
        # read json file
        json_data = self.read_json(json_filename)
        # get insulin sensitivity
        self.insulin_sensitivity = self.set_insulin_sensitivity(json_data)
        # get carb ratio
        self.carb_ratio = self.set_carb_ratio(json_data)
        # get id from filename
        self.id = self.set_id(filename)

    # return insulin sensitivity
    def get_insulin_sensitivity(self) -> float:
        if self.insulin_sensitivity is None:
            logging.error("Insulin Sensitivity is not set")
            raise Exception("insulin sensitivity not set")
        return self.insulin_sensitivity

    # return carb ratio
    def get_carb_ratio(self) -> float:
        if self.carb_ratio is None:
            logging.error("Carb ratio is not set")
            raise Exception("Carb ratio is not set")
        return self.carb_ratio
    
    # return Id
    def get_id(self) -> str:
        return self.id


    # convert filename
    def convert_filename(self, filename:str) -> str:
        logging.info("convert filename {}".format(filename))
        # set suffix to json
        new_filename = filename[:-3] + 'json'
        # replace csv with profile
        new_filename = new_filename.replace('csv', 'profile')
        return new_filename

    # read json file
    def read_json(self, filename:str) -> {}:
        logging.info("reading json file {}".format(filename))
        with open(filename, 'r') as f:
            profile = json.load(f)
        return profile[0]

    # extract insulin sensitivity from json
    def set_insulin_sensitivity(self, json_data: {}):
        logging.info("get insulin sensitivity")
        sens_array = json_data['store']['Default']['sens']
        if len(sens_array) > 1:
            logging.error("More than one sensitivity reading")
        sensitivity = sens_array[0]['value']
        return sensitivity

    # extract carb ratio from json
    def set_carb_ratio(self, json_data: {}):
        logging.info("get carb ratio")
        carb_ratio_array = json_data['store']['Default']['carbratio']
        if len(carb_ratio_array) > 1:
            logging.error("More than one carb ratio reading")
        carb_ratio = carb_ratio_array[0]['value']
        return carb_ratio

    # extract id from filename
    def set_id(self, filename: str):
        logging.info("extract id from filename")
        regex = r"(?<=csv_)\d+(?=\.csv)"
        matches = re.findall(regex, filename)
        logging.info("id: {}".format(matches[0]))
        return matches[0]