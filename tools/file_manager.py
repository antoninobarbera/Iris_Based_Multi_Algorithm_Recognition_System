import os
import sys
import shutil
from pathlib import Path
import yaml
from addict import Dict
from yaml_config_override import add_arguments
import pickle


def directory_exists(path):
    '''
       directory_exists() checks if a directory truly exists. If it does not exist, the program will be terminated.

         Args:
         path: The directory path.
         message: The Error Message that will be printed before the program terminates. 

        Returns:
        True: if the directory exists.
    '''
    if os.path.isdir(path):
        return True  
    else:
        return False
        
def file_exists(file_path, message):
    '''
       file_exists() checks if a file truly exists. If it does not exist, the program will be terminated.

         Args:
         path: The file path.
         message: The Error Message that will be printed before the program terminates. 

        Returns:
        True: if the file exists.
    '''
    if os.path.isfile(file_path):
        return True
    else: 
        print(message)
        sys.exit()

def move_directory(source, destination):
    '''
       move_directory() moves a directory into another directory.

         Args:
         source: The path of the directory that will be moved.
         destination: The destination directory path.
    '''
    path = os.path.join(destination, source)
    if os.path.isdir(path): 
        shutil.rmtree(path)
    shutil.move(source, destination)

    
def configuration(main_path=None):
    '''
       configuration() extract from the configuration file (config\base_config.yaml) the configuration parameters
       and save it in a Dict object. Also it checks if the configuration file exists.

        Args:
        main_path(path string): The path of the parent folder. Default value: None. 

        Returns:
        config (addict Dict): Dict containing configuration parameters.
    '''

    # Set path.
    relative_path = os.path.join('config', 'base_config.yaml')
    if main_path is None: 
        path = relative_path
    else:
        path = os.path.join(main_path, relative_path)

    file_exists(path, " NO CONFIGURATION FILE DETECTED --> ADD A base_config.yaml FILE TO config DIRECTORY ")
    config_source = yaml.safe_load(Path(path).read_text())
    config = Dict(add_arguments(config_source))
    return config

def load_dataset(config):
    '''
    Loads the CASIA dataset from the 'dataset' directory.
    
    Args:
    - config (addict.Dict): Configuration object containing parameters.
    
    Returns:
    - list: A list of records from the CASIA dataset.
    '''
    dataset_path = 'dataset'
    casia_path = os.path.join(dataset_path, 'CASIA.pkl')
        
    if not os.path.exists(dataset_path):
        print(f"Error: The folder '{dataset_path}' does not exist.")
        sys.exit()

    if not os.path.isfile(casia_path):
        print(f"Error: The file '{casia_path}' does not exist.")
        sys.exit()
        
    casia_path = os.path.join('dataset', 'CASIA.pkl')
        
    try:
        with open(casia_path, 'rb') as file:
            casia = pickle.load(file)
    except Exception as e:
        print(f"ERROR trying to load '{casia_path}'.")
        print(f"Exception details: {str(e)}")
        sys.exit()

    rec_1 = casia['rec_1']
    rec_2 = casia['rec_2']
    rec_3 = casia['rec_3']
    rec_4 = casia['rec_4']
    rec_5 = casia['rec_5']
    rec_6 = casia['rec_6']
    rec_7 = casia['rec_7']

    return [rec_1, rec_2, rec_3, rec_4, rec_5, rec_6, rec_7]
