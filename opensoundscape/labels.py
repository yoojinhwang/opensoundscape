import pandas as pd
import numpy as np
import os
import csv
import re 

# name_table_path = '/lacie/projects1/Sam-Lapp/OPSO/output/translation_table.csv'
# print(f'loading name translation table from {name_table_path}')
# name_table = pd.read_csv(name_table_path)
# name_table = name_table.drop_duplicates(subset='com-name')
# name_table = name_table.reset_index(drop=True)
# print(f'number of species in translation table: {len(name_table)}')

def clean(mystr):
    """remove all characters except alphanumeric, and convert to lowercase"""
    return re.sub(r"[^a-zA-Z0-9]","",mystr).lower()

#raven 
def find_species_in_time_segment(raven_label_df,start_time,end_time,min_overlap=0.5):
    #start_time and end_time are the start/end of the clip. 
    #we want events that start earlier than the end of the clip,
    #and end later than the beginning of the clip
    start_time_offset = start_time + min_overlap
    end_time_offset = end_time - min_overlap
    
    #subset the rows of the DF for this clip
    #first subset by requiring events to start before the end of the clip
    subset_df = raven_label_df[raven_label_df['Begin Time (s)'] < end_time_offset]
    subset_df = subset_df[subset_df['End Time (s)'] > start_time_offset]
    
    species_in_clip = subset_df.Species.tolist()
    
    return species_in_clip

def species_in_alpha_list(species, alpha_codes):
    if not (species in lookup_alphacode.index.values):
        return 0
    elif lookup_alphacode.loc[species].code in alpha_codes:
        return 1
    else:
        return 0

def createSpeciesPresenceVector(alpha_codes,species_list):
    binary_presence = [species_in_alpha_list(s, alpha_codes) for s in species_list]
    return binary_presence

def raven_annotations_to_labels(df, min_presence_time = 0.5, annotation_column='Species')
    """df should contain audio_path, start_time, end_time, raven_txt_file"""
    unique_raven_files = np.unique(df.raven_txt_file)
    
    for raven_txt_file in unique_raven_files:
        #load one raven annotation file and process all clips associated with it
        raven_labels = pd.read_csv(raven_txt_file, delimiter='\t')
        
         #if Species column is named "Annotation", rename it to "Species":
        if not ('Species' in raven_labels.columns):
            raven_labels.columns = [c if c!='Annotation' else 'Species' for c in labelDF.columns]
        
        associated_clips = df[df['raven_txt_file']==raven_txt_file]
        for i, row in associated_clips.iterrows():
            species = find_species_in_time_segment(raven_labels,row.start_time,row.end_time,min_overlap=minimum_presence_time)
            unique_species = np.unique(species)
            label_df.loc[row.audio_path] = []
        

def create_bn_to_xc_dictionary(species_table_path='/lacie/projects1/Sam-Lapp/OPSO/resources/species_table.csv'):
    """returns {bn_code : xc_sci_name} dictionary and ordered [species_list], from species table"""
    
    print(f'reading species translations from table at {species_table_path}')
    
    #load Tessa's species translation table
    species_table = pd.read_csv(species_table_path)

    #create a dictionary that maps from 6 letter bn codes to xc scientific name as lowercase-hyphenated
    bn_to_xc = {}
    for i, row in species_table.iterrows():
        bn_code = row.bn_code
        xc_sci_name = row.bn_mapping_in_xc_dataset
        #if both columns have a value, make a key-value pair in dictionary
        if bn_code is not np.nan and xc_sci_name is not np.nan:
            bn_to_xc[bn_code] = xc_sci_name
    xc_species_list = list(np.sort(list(bn_to_xc.values())))

    print(f'number of species in species table: {len(xc_species_list)}')
    return bn_to_xc, xc_species_list

bn_to_xc_dict, species_list_sci = create_bn_to_xc_dictionary()

def get_species_list():
    return species_list_sci

def birdNetTables_to_speciesDF(table_paths, audio_paths): #, species_list_sci):
    n_obs = len(audio_paths)
    n_classes = len(species_list_sci)
    
    predictionDF = pd.DataFrame(np.zeros([n_obs,n_classes]),columns=species_list_sci)
    predictionDF['fullpath'] = audio_paths
    predictionDF['file'] = [os.path.basename(p) for p in audio_paths]
    predictionDF = predictionDF.set_index('file',drop=True)
    
    for table_path in table_paths:
        filename = '' #we will read it from the prediction table
       
        with open(table_path, newline='') as table:
            rows = csv.reader(table, delimiter='\t')#, quotechar='|')
            for i,row in enumerate(rows):
                #reading one row of the birdNet prediction table for one file. 
                #It contains a confidence for a single species in a 3-second segment
                if i==0:
                    continue
                if i==1:
                    filename = row[3]
                
#                 com_name = row[-2]
#                 sci_name = sci_name_bn[com_name]
                bn_code = row[-3]
                

                try:
                    sci_name = bn_to_xc_dict[bn_code]
                except KeyError:
                    continue #we don't have this species in our lookup table, so we ignore it
#                 if sci_name is np.nan: #we don't have this species in our naming lookup table
#                     continue
                    
                confidence = float(row[-1])
                
                #update the confidence for this species in this file, if it is higher than the stored value
#                 if sci_name is not np.nan: #because this is faster than np.nanmax()
                predictionDF.at[filename,sci_name] = max(predictionDF.at[filename,sci_name],confidence)
    return predictionDF

# #for example:
# from glob import glob
# import pandas as pd
# from opso.birdnet_tools import birdNetTables_to_speciesDF

# birdnet_prediction_tables = glob('/Volumes/seagate5/birdclef_2019_val/birdnet_prediction_tables/*BirdNET.selections.txt')
# print(len(birdnet_prediction_tables))
# audio_paths = glob('/Volumes/seagate5/birdclef_2019_val/audio-SPLIT/*.mp3')
# print(len(audio_paths))
# species_list_sci = pd.read_csv("./output/species_list.txt").columns.tolist()
# print(len(species_list_sci))
# birdNet_ithaca_predictions_df = birdNetTables_to_speciesDF(birdnet_prediction_tables[0:10], audio_paths, species_list_sci)
# birdNet_ithaca_predictions_df.head()

