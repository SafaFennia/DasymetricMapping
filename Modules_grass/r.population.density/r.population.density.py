#!/usr/bin/env python
# encoding: utf-8
#****************************************************************
#*
#* MODULE:     r.population.density
#*
#* AUTHOR(S):  Grippa Tais, Safa Fennia
#*
#* PURPOSE:    Create a weighting layer for dasymetric mapping, using a random forest regression model
#*
#* COPYRIGHT:  (C) 2017 Grippa Tais, Safa Fennia
#*             ANAGEO, Université libre de Bruxelles, Belgique
#*
#*
#*             First Version: 2017/07/18
#*             Second Version: 2017/11/13
#*             Third Version: 2018/02/02
#*
#*             This program is free software under the
#*             GNU General Public License (>=v2).
#*             Read the file COPYING that comes with GRASS
#*             for details.
#*
#****************************************************************

#%module
#% description: Creates a weighting layer with Random Forest for population redistribution (dasymetric mapping)
#% keywords: Land Cover, Density, Population, Dasymetry, Random Forest, Dasymetric mapping
#%end
#%option G_OPT_V_INPUT
#% key: vector
#% description: Input Vector: Vector with administrative zones (polygon) with unique ID and population count contained in the attribute table
#% required : yes
#%end
#%option G_OPT_R_INPUT
#% key: land_cover
#% type: string
#% description: Input raster: Land Cover
#% required : yes
#%end
#%option G_OPT_R_INPUT
#% key: land_use
#% type: string
#% description: Input raster: Land use, morphological areas...
#% required : no
#%end
#%option G_OPT_R_INPUT
#% key: distance_to
#% type: string
#% description: Input raster: Distance to zones of interest
#% required : no
#%end
#%option
#% key: tile_size
#% key_desc: value
#% type: Integer
#% description: Spatial resolution (in meters) of the weighting layer to be produced
#% required : yes
#%end
#%option
#% key: id
#% type: string
#% description: Name of the column containing the ID of administrative units (unique identifier)
#% required : yes
#%end
#%option
#% key: population
#% type: string
#% description: Name of the column containing the population count
#% required : yes
#%end
#%option G_OPT_V_CAT
#% key: built_up_category
#% type: Integer
#% description: Built_up_category
#% required : no
#%end
#%option G_OPT_R_OUTPUT
#% key: output
#% description: Name for output weighting layer
#% required: yes
#% guisection: Output
#%end
#%option G_OPT_F_OUTPUT
#% key: plot
#% description: Name for output plot of model's feature importances
#% required: yes
#% guisection: Output
#%end
#%option G_OPT_F_OUTPUT
#% key: log_file
#% description: Name for output file with logging of the random forest run
#% required: no
#% guisection: Output
#%end
#%option
#% key: lc_list
#% type: string
#% label: Land cover's categories to be used
#% description: Format: 1 2 3 thru 7 *
#% required: no
#%end
#%option
#% key: lu_list
#% type: string
#% label: Land use's categories to be used
#% description: Format: 1 2 3 thru 7 *
#% required: no
#%end
#%option G_OPT_F_INPUT
#% key: lc_class_name
#% description: Csv file containing class names for land cover map
#% required: no
#%end
#%option G_OPT_F_INPUT
#% key: lu_class_name
#% description: Csv file containing class names for land use map
#% required: no
#%end
#%option
#% key: n_jobs
#% type: integer
#% description: Number of cores to be used for the parallel process
#% required : yes
#%end
#%flag
#% key: a
#% description: Keep all covariates in the final model
#% guisection: Feature selection and tuning
#%end
#%flag
#% key: f
#% description: Include detailed results of grid search cross-validation
#% guisection: Feature selection and tuning
#%end
#%option
#% key: kfold
#% type: integer
#% description: Number of k-fold cross-validation for grid search parameter optimization
#% required: no
#% guisection: Feature selection and tuning
#%end
#%option
#% key: param_grid
#% type: string
#% description: Python dictionary of customized tunegrid for sklearn RFregressor
#% required: no
#% guisection: Feature selection and tuning
#%end
#%rules
#% required: land_cover, land_use
#% requires: lc_list, land_cover
#% requires: lu_list, land_use
#% requires: lc_class_name, land_cover
#% requires: lu_class_name, land_use
#% requires: -f, log_file
#%end


## Import standard python libraries
import os, sys, glob, time
import csv
import atexit
## Import GRASS GIS Python Scripting Library
import grass.script as gscript
## Import Shutil library
import shutil
## Import Numpy library
import numpy as np
## import math library
import math as ma
## import pyplot library
import matplotlib
matplotlib.use('Agg') #use a non-interactive backend: prevent the figure from popping up
import matplotlib.pyplot as plt
# import multiprocessing and functools libraries
import multiprocessing
from multiprocessing import Pool
from functools import partial 
# import literal_eval
from ast import literal_eval

## Import Pandas library (View and manipulaiton of tables)
try:
    import pandas as pd  #TODO: remove dependency to Pandas (still requiered for selecting in dataframes based on column names)
except:
    gscript.fatal("Pandas is not installed ")

## Import sklearn libraries
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import SelectFromModel
    from sklearn.model_selection import GridSearchCV
except:
    gscript.fatal("Scikit learn 0.18 or newer is not installed")


def cleanup():
    gscript.run_command('g.remove', quiet=True, type='raster', name=','.join(TMP_MAPS), flags='fb')
    for tmp_csv in TMP_CSV:
        if os.path.isfile(tmp_csv):
            os.remove(tmp_csv)

def create_tempdirs():
    '''
    Function that create needed temporary folder. Those name have to be saved as other function will depend of the name of those folder.
    '''
    # Temporary directory for administrative units statistics
    global outputdirectory_admin
    tmp_grass_dir=gscript.tempdir()
    outputdirectory_admin=os.path.join(tmp_grass_dir,"admin_level")
    if not os.path.exists(outputdirectory_admin):
        os.makedirs(outputdirectory_admin)
    # Temporary directory for grids statistics
    global outputdirectory_grid
    outputdirectory_grid=os.path.join(tmp_grass_dir,"grid_level")
    if not os.path.exists(outputdirectory_grid):
        os.makedirs(outputdirectory_grid)

def Data_prep(categorical_raster):
    '''
    Function that extracts resolution and sorted list of classes of a categorical raster (like land cover or land use information).
    '''
    info = gscript.raster_info(categorical_raster)
    nsres=info.nsres
    ewres=info.ewres
    L = []
    L=[cl for cl in gscript.parse_command('r.category',map=categorical_raster)]
    for i,x in enumerate(L):  #Make sure the format is UTF8 and not Unicode
        L[i]=x.encode('UTF8')
    L.sort(key=float) #Sort the raster categories in ascending.
    return nsres, ewres, L

def category_list_check(cat_list, raster_map):
    '''
    Function checking if the category provided via a list well exists in the corresponding raster_map
    '''
    existing_cat=Data_prep(raster_map.split("@")[0])[2]
    for cat in cat_list:
        if cat not in existing_cat:
            gscript.fatal(_("Some categories provided via a list does not exists in <%s> raster. Please check.") % raster_map)

def check_no_missing_zones(vector_origin, vector_gridded):
    '''
    Function checking if the number of items (admin zones) in the original vector provided by the user is wall conserved after the rasterization.
    If the original vector contains small sized polygons (or very tight) and desired 'tile_size' is too large, some polygons could disappeared during the rasterization process
    '''
    origin_n=gscript.parse_command('v.db.univar', flags='g', map=vector_origin, column='cat')['n']
    gridded_n=gscript.parse_command('v.db.univar', flags='g', map=vector_gridded, column='cat')['n']
    if origin_n != gridded_n:
        gscript.run_command('g.remove', quiet=True, type='vector', name=vector_gridded, flags='fb')
        message=_(("A tile size of %s m seems to large and produce loss of some administrative units when rasterizing them.\n") % tile_size)
        message+=_(("Try to reduce the 'tile_size' parameter or edit the <%s> vector to merge smallest administrative units with their neighoring units") % vector_origin)
        gscript.fatal(message)
    
    
def admin_boundaries(vector, id):
    '''
    Function convecting the vecotor to raster then raster to vector: boundaries will have a staircase appearence
    so that each tile of the gridded vector will be contained in only one administrative unit
    '''
    global gridded_vector
    current_mapset=gscript.gisenv()['MAPSET']
    gridded_vector=vector.split("@")[0]+'_'+str(tile_size)+'m_gridded'+'@'+current_mapset
    gscript.run_command('g.region', raster='clumped_grid')
    gscript.run_command('v.to.rast', quiet=True, input=vector, type='area', output='gridded_admin_units', use='attr', attribute_column=id, overwrite=True)
    gscript.run_command('r.to.vect', quiet=True, input='gridded_admin_units', output=gridded_vector, type='area', column=id, flags='v',overwrite=True)
    temp_name=random_string(15)
    gscript.run_command('g.copy', quiet=True, vector='%s,%s'%(vector,temp_name))
    gscript.run_command('v.db.join', quiet=True, map_=gridded_vector, column='cat', other_table=temp_name, other_column=id, subset_columns=population) #join the population count
    gscript.run_command('g.remove', quiet=True, flags='f', type='vector', name=temp_name+'@'+current_mapset)
    TMP_MAPS.append("gridded_admin_units")
    check_no_missing_zones(vector,gridded_vector)
    
    
def create_clumped_grid(tile_size):
    '''
    Function creating clumped grid which will be used for computing raster's classes proportion at grid level. This clumped grid will
    also be used at the end with r.reclass to allow random forest prediction to each grid
    '''
    gscript.run_command('g.region', raster=Land_cover.split("@")[0], res=tile_size)
    gscript.run_command('r.mask', quiet=True, raster=Land_cover.split("@")[0])
    gscript.mapcalc("empty_grid=rand(0 ,999999999)", overwrite=True, seed='auto') #Creating a raster with random values
    gscript.run_command('r.clump', quiet=True, input='empty_grid',output='clumped_grid',overwrite=True) #Assigning a unique value to each grid
    gscript.run_command('r.mask', quiet=True, flags='r')
    TMP_MAPS.append("empty_grid")
    TMP_MAPS.append("clumped_grid")
    
    
def random_string(N):
    '''
    Function generating a random string of size N
    '''
    import random, string
    prefix=random.choice(string.ascii_uppercase + string.ascii_lowercase)
    suffix=''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(N))
    return prefix+suffix
    
    
def proportion_class(rasterLayer, cl):
    '''
    Function extracting a binary map for class 'cl' in raster 'rasterLayer', then computing the proportion of this class in both administratives units and in grids. 
    The computational region should be defined properly before running this function. 
    '''
    ### Create a binary raster for the current class
    prefix = 'LC' if rasterLayer == Land_cover.split("@")[0] else 'LU'  # Adaptative prefix according to the input raster (land_cover of land_use)
    binary_raster = prefix+"_"+cl  # Set the name of the binary raster
    gscript.run_command('r.mapcalc', expression='%s=if(%s==%s,1,0)'%(binary_raster,rasterLayer,cl),overwrite=True,quiet=True) # Mapcalc to create binary raster for the expected class 'cl'
    ### Create a temporary copy of the current binary raster with all pixels values equal to 1 (to be used for computing proportion of current binary class)
    tmplayer='tmp_%s_%s'%(binary_raster,random_string(4))
    gscript.run_command('r.mapcalc', expression='%s=if(%s==1,1,1)'%(tmplayer,binary_raster),overwrite=True,quiet=True) # Mapcalc to create binary raster for the expected class 'cl'
    # Fill potential remaining null values with 0 value (null values existing in the 'rasterLayer' will remain null in the binary, using r.mapcalc)
    gscript.run_command('r.null', quiet=True, map=binary_raster, null='0') 
    gscript.run_command('r.null', quiet=True, map=tmplayer, null='0')   
    ### Compute proportion of pixels of the current class - Administrative units
    stat_csv=os.path.join(outputdirectory_admin,"%s_%s.csv"%(prefix,cl))
    ref_map='gridded_admin_units'
    gscript.run_command('i.segment.stats', flags='s', map=ref_map, rasters='%s,%s'%(tmplayer,binary_raster), raster_statistics='sum', csvfile=stat_csv, separator='comma', quiet=True, overwrite=True)
    output_csv_1=compute_proportion_csv(stat_csv) #Create a new csv containing the proportion
    ### Compute proportion of pixels of the current class - Grids
    stat_csv=os.path.join(outputdirectory_grid,"%s_%s.csv"%(prefix,cl))
    ref_map='clumped_grid'
    gscript.run_command('i.segment.stats', flags='s', map=ref_map, rasters='%s,%s'%(tmplayer,binary_raster), raster_statistics='sum', csvfile=stat_csv, separator='comma', quiet=True, overwrite=True)
    output_csv_2=compute_proportion_csv(stat_csv) #Create a new csv containing the proportion
    ### Remove temporary layer
    gscript.run_command('g.remove', quiet=True, flags='f',type='raster',name=tmplayer) 
    # Return lists
    return (binary_raster,output_csv_1,output_csv_2)
    
    
def compute_proportion_csv(infile):
    '''
    Function used in 'proportion_class' function. It take as input the csv from i.segment.stats with the area (in number of pixels)
    the sum of pixels of the binary raster and create a new csv with the proportion
    '''
    # Set the path to the outputfile
    head, tail = os.path.split(infile)
    root, ext = os.path.splitext(tail)
    outfile=os.path.join(head,root+"_prop"+ext)
    # Create new csv reader and writer objects
    reader=csv.reader(open(infile,'r'), delimiter=",")
    writer=csv.writer(open(outfile,'w'), delimiter=",")
    # Initialize empty lists
    crash_report=[]
    content=[]
    # Save the first line as header and create the new header
    header=reader.next()
    new_header=[]
    new_header.append(header[0])
    index=header[2].find("_sum")
    new_header.append(header[2][:index]+'_proportion')
    content.append(new_header)  #Create new header with first original column and current class related name for proportion
    # Loop through the rest of the rows (header is passed)
    for row in reader:
        pix_nb=float(row[1]) #Area of the unit (in number of pixels)
        class_nb=float(row[2]) #Number of pixels of current class (binary raster)
        try:
            prop=100*class_nb/pix_nb
            content.append([row[0],"{0:.5f}".format(prop)])
        except ZeroDivisionError:  #If computation of proportion failed because of 'ZeroDivisionError'
            crash_report.append(row[0])
            content.append([row[0],"{0:.5f}".format(0.0)])  # If ZeroDivisionError, set the proportion to zero to avoid errors in next steps
            continue
    writer.writerows(content)
    os.remove(infile)
    # Print notification of ZeroDivisionError if it happened
    if len(crash_report)>0:
        print "An 'ZeroDivisionError' has been registered for the following <%s>"%header[0]+"\n".join(crash_report)
    # Return the path to the temporary csv file
    return outfile


def atoi(text):
    '''
    Function that return integer if text is digit - Used in 'natural_keys' function
    '''
    return int(text) if text.isdigit() else text


def natural_keys(text):   #Return key to be used for sorting string containing numerical values - Used in 'join_csv' function
    '''
    Trick was found here
    https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)

    '''
    import re  #Import needed library
    return [ atoi(c) for c in re.split('(\d+)', text) ]  #Split the string


def ordered_list_of_path(indir,pattern_A,pattern_B="",pattern_C=""):
    '''
    Function that return a list of ordered path for the files in the folder 'indir'.
    'pattern_A', 'pattern_B', 'pattern_C'
    '''
    # Make a list of .csv files according to their filename pattern
    os.chdir(indir) # Change the current directory to the folder containing all the .csv files
    csvList=glob.glob(pattern_A) #Make a list of strings with the name of .csv files
    csvList.sort(key=natural_keys) #Sort the list on a human natural order (for strings containing numericals)
    if pattern_B !="":
        csvList_B=glob.glob(pattern_B) #Make a list of strings with the name of .csv files
        csvList_B.sort(key=natural_keys) #Sort the list on a human natural order (for strings containing numericals)
        for item in csvList_B:
            csvList.append(item)
    if pattern_C !="":
        csvList_C=glob.glob(pattern_C) #Make a list of strings with the name of .csv files
        csvList_C.sort(key=natural_keys) #Sort the list on a human natural order (for strings containing numericals)
        for item in csvList_C:
            csvList.append(item)
    return csvList

            
def join_2csv(file1,file2,separator=";",join='inner',fillempty='NULL'):
    '''
    Function that join two csv files according to the first column (primary key).
    'file1' and 'file2' wait for complete path (strings) to the corresponding files. Please not that 'file1' is assume to be the left-one in the join
    'separator' wait for the character to be considered as .csv delimiter (string)
    'join' parameter wait either for 'left' or 'inner' according to type of join
    'fillempty' wait for the string to be use to fill the blank when no occurance is found for the join operation
    '''
    import time,csv,os
    header_list=[]
    file1_values_dict={}
    file2_values_dict={}
    reader1=csv.reader(open(file1), delimiter=separator) #Csv reader for file 1
    reader2=csv.reader(open(file2), delimiter=separator) #Csv reader for file 2
    # Make a list of headers
    header_list1=[ x for x in reader1.next()]
    header_list2=[ x for x in reader2.next()[1:]]
    # Make a list of unique IDs from the first and second table according to type of join
    if join=='inner':
        id_list=[row[0] for row in reader1]
        [id_list.append(row[0]) for row in reader2]
        id_list=list(set(id_list))
        id_list.sort(key=natural_keys)
    if join=='left':
        id_list=[row[0] for row in reader1]
        id_list=list(set(id_list))
        id_list.sort(key=natural_keys)
    # Build dictionnary for values of file 1
    reader1=csv.reader(open(file1), delimiter=separator)
    reader1.next()
    values_dict1={rows[0]:rows[1:] for rows in reader1}
    # Build dictionnary for values of file 2
    reader2=csv.reader(open(file2), delimiter=separator)
    reader2.next()
    values_dict2={rows[0]:rows[1:] for rows in reader2}
    # Built new content
    new_content=[]
    new_header=header_list1+header_list2
    new_content.append(new_header)
    for key in id_list:
        new_row=[key]
        try:
            [new_row.append(value) for value in values_dict1[key]]
        except:
            [new_row.append('%s'%fillempty) for x in header_list1[1:]]
        try:
            [new_row.append(value) for value in values_dict2[key]]
        except:
            [new_row.append('%s'%fillempty) for x in header_list2]
        new_content.append(new_row)
    #Return the result
    outfile=gscript.tempfile()
    fout=open(outfile,"w")
    writer=csv.writer(fout, delimiter=separator)
    writer.writerows(new_content) #Write multiples rows in the file
    time.sleep(0.5) # To be sure the file will not be close to fast (the content could be uncompletly filled) 
    fout.close()
    return outfile

def join_multiplecsv(fileList,outfile,separator=";",join='inner', fillempty='NULL', overwrite=False):
    '''
    Function that apply join on multiple csv files
    '''
    import os, sys, shutil
    # Stop execution if outputfile exitst and can not be overwriten
    if os.path.isfile(outfile) and overwrite==False:
        print "File '%s' aleady exists and overwrite option is not enabled."%outfile
    else:
        if os.path.isfile(outfile) and overwrite==True:  # If outputfile exitst and can be overwriten
            os.remove(outfile)
            #print "File '%s' will be overwrited."%outfile   # Uncomment if you want a print
        nbfile=len(fileList)
        if nbfile<=1: #Check if there are at least 2 files in the list
            sys.exit("This function require at least two .csv files to be jointed together.")
        # Copy the list of file in a queue list
        queue_list=list(fileList)
        # Left join on the two first files
        file1=queue_list.pop(0)
        file2=queue_list.pop(0)
        tmp_file=join_2csv(file1,file2,separator=separator,join=join, fillempty=fillempty)
        # Left join on the rest of the files in the list
        while len(queue_list)>0:
            file2=queue_list.pop(0)
            tmp_file=join_2csv(tmp_file,file2,separator=separator,join=join, fillempty=fillempty)
        #Copy the temporary file to the desired output path
        shutil.copy2(tmp_file,outfile)
        # Print what happend
        #print "%s individual .csv files were joint together."%nbfile    # Uncomment if you want a print
        

def labels_from_csv(current_labels):
    '''
    Function that take as input a list of current labels for 'LC' and 'LU' classes and return a list of modified
    labels according to the classes' names provided via "lc_class_name" and "lu_class_name"
    '''
    new_label=[]
    lc_class_rename_dict={}
    lu_class_rename_dict={}
    if lc_class_name != "":
        for row in open(lc_class_name):
            classcode,classname=row.replace('\r\n','\n').split('\n')[0].split('|')
            lc_class_rename_dict[classcode]=classname
    if lu_class_name != "":
        for row in open(lu_class_name):
            classcode,classname=row.replace('\r\n','\n').split('\n')[0].split('|')
            lu_class_rename_dict[classcode]=classname
    for l in current_labels:
        if l[:2]=='LC':
            classnum=l[3:l.index('_proportion')]
            if classnum in lc_class_rename_dict.keys():
                new_label.append('LC "'+lc_class_rename_dict[classnum]+'"')
            else:
                new_label.append('LC "'+classnum+'"')
        elif l[:2]=='LU':
                        classnum=l[3:l.index('_proportion')]
                        if classnum in lu_class_rename_dict.keys():
                            new_label.append('LU "'+lu_class_rename_dict[classnum]+'"')
                        else:
                            new_label.append('LU "'+classnum+'"')
        elif l[-4:]=='mean':
            new_label.append(l[0].title()+l[1:-5].replace('_',' '))
        else:
            new_label.append(l)
    return new_label

def RandomForest(weigthing_layer_name,vector,id):
    '''
    Function that creates a random forest model trained at the administrative units level to generate gridded prediction
    covariates are proportion of each Land Cover's class (opt: with proportion of each land use's class)
    '''
    global log_text, log_text_extend, n_jobs, kfold
    # -------------------------------------------------------------------------
    # Data preparation for administrative units
    # -------------------------------------------------------------------------
    # Compute area of the gridded administrative unit (vector) layer (+add column)
    gscript.run_command('v.db.addcolumn', quiet=True, map=gridded_vector, columns="area double precision")
    gscript.run_command('v.to.db', quiet=True, map=gridded_vector, option='area', columns='area', units='meters')
    # Export desired columns from the attribute table as CSV
    tmp_table=os.path.join(outputdirectory_admin,"tmp_%s.csv"%random_string(4)) # Define the path to the .csv
    query="SELECT cat,%s,area FROM %s"%(population,gridded_vector.split('@')[0])
    gscript.run_command('db.select', quiet=True, sql=query, output=tmp_table)
    TMP_CSV.append(tmp_table)
    # Compute log of density in a new .csv file
    reader=csv.reader(open(tmp_table,'r'), delimiter='|')
    log_density_csv=os.path.join(outputdirectory_admin,"log_pop_density.csv") # Define the path to the .csv containing the log of density
    fout=open(log_density_csv,'w')
    writer=csv.writer(fout, delimiter=',')
    new_content=[]
    new_header=['cat','log_population_density']
    new_content.append(new_header)
    reader.next() # Pass the header
    [new_content.append([row[0],ma.log(int(row[1])/float(row[2]))]) for row in reader]  # Compute log (ln) of the density
    writer.writerows(new_content)
    time.sleep(0.5) # To be sure the file will not be close to fast (the content could be uncompletly filled) 
    fout.close()
    # Define the path to the file with all co-variates 
    all_stats_grid=os.path.join(outputdirectory_grid,"all_stats.csv") # for grid level
    all_stats_admin=os.path.join(outputdirectory_admin,"all_stats.csv") # for admin level
    # For admin level : join all co-variates with the log of density (response variable of the model) 
    tmp_file=join_2csv(log_density_csv,all_stats_admin,separator=",",join='inner',fillempty='NULL')        
    admin_attribute_table=os.path.join(outputdirectory_admin,"admin_attribute_table.csv")
    shutil.copy2(tmp_file,admin_attribute_table) # Copy the file from temp folder to admin folder
    TMP_CSV.append(tmp_file)
    
    # -------------------------------------------------------------------------
    # Creating RF model
    # -------------------------------------------------------------------------
    df_admin = pd.read_csv(admin_attribute_table) #reading the csv file as dataframe
    df_grid = pd.read_csv(all_stats_grid)

    ## Changing null values to zero
    # for df_grid
    #features = df_grid.columns[:]
    #for i in features:
    #    df_grid[i].fillna(0, inplace=True)
    # for df_admin
    #features = df_admin.columns[:]
    #for i in features:
    #    df_admin[i].fillna(0, inplace=True)

    ## Make a list with name of covariables columns
    list_covar=[]
    for cl in lc_classes_list:
        list_covar.append("LC_"+cl+"_proportion")
    if (Land_use != ''):
        for cl in lu_classes_list:
            list_covar.append("LU_"+cl+"_proportion")
    if(distance_to != ''):
        list_covar.append(distance_to.split("@")[0]+"_mean")

    ## Saving variable to predict (dependent variable)
    y = df_admin['log_population_density']

    ## Saving covariable for prediction (independent variables)
    x=df_admin[list_covar]  #Get a dataframe with independent variables for administratives units
    #x.to_csv(path_or_buf=os.path.join(outputdirectory_admin,"covar_x.csv"), index=False) #Export in .csv for archive

    # Remove features whose importance is less than a threshold (Feature selection)
    rfmodel=RandomForestRegressor(n_estimators = 500, oob_score = True, max_features='auto', n_jobs=-1)
    a=SelectFromModel(rfmodel, threshold=min_fimportance)
    fited=a.fit(x, y)
    feature_idx = fited.get_support()   # Get list of True/False values according to the fact the OOB score of the covariate is upper the threshold 
    list_covar = list(x.columns[feature_idx])  # Update list of covariates with the selected features 
    x=fited.transform(x)  # Replace the dataframe with the selected features
    message="Selected covariates for the random forest model (with feature importance upper than {value} %)".format(value=min_fimportance*100)  # Print the selected covariates for the model
    message+="\n".join(list_covar)
    log_text+=message+'\n'
    
    #### Tuning of hyperparameters for the Random Forest regressor using "Grid search"
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, cv=kfold, n_jobs=n_jobs, verbose=0)
    grid_search.fit(x, y)   # Fit the grid search to the data
    regressor = grid_search.best_estimator_  # Save the best regressor
    regressor.fit(x, y) # Fit the best regressor with the data
    
    # Print infos and save it in the logfile - Grid of parameter to be tested
    message='Parameter grid for Random Forest tuning :\n'
    for key in param_grid.keys():
        message+='    '+key+' : '+', '.join([str(i) for i in list(param_grid[key])])+'\n'
    log_text+=message+'\n'
    print message
    # Print infos and save it in the logfile - Tuned parameters
    message='Optimized parameters for Random Forest after grid search %s-fold cross-validation tuning :\n'%kfold
    for key in grid_search.best_params_.keys():
        message+='    %s : %s'%(key,grid_search.best_params_[key])+'\n'
    log_text+=message+'\n'
    print message
    # Print info of the mean cross-validated score (OOB) and stddev of the best_estimator
    best_score=grid_search.cv_results_['mean_test_score'][grid_search.best_index_]
    best_std=grid_search.cv_results_['std_test_score'][grid_search.best_index_]
    message="Mean cross-validated score (OOB) and stddev of the best_estimator : %0.3f (+/-%0.3f)"%(best_score,best_std)+'\n'
    log_text+=message+'\n'
    print message
    # Print mean OOB and stddev for each set of parameters
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    message="Mean cross-validated score (OOB) and stddev for every tested set of parameter :\n"
    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        message+="%0.3f (+/-%0.03f) for %r"% (mean, std, params)+'\n'
    log_text_extend+=message
    
    # Predict on grids
    x_grid=df_grid[list_covar] #Get a dataframe with independent variables for grids (remaining after feature selection)
    #x_grid.to_csv(path_or_buf=os.path.join(outputdirectory_grid,"covar_x_grid.csv"), index=False) #Export in .csv for archive
    prediction = regressor.predict(x_grid)  # Apply the model on grid values

    # Save the prediction
    df1 = df_grid['cat']
    df2 = pd.DataFrame(prediction, columns=['log'])
    df_weight = pd.concat((df1,df2), axis=1)
    col = df_weight.apply(lambda row : np.exp(row["log"]), axis=1)
    df_weight ["weight_after_log"] = col
    weightcsv=os.path.join(outputdirectory_grid,"weight.csv")
    #df_weight.to_csv(path_or_buf=weightcsv) #Export in .csv for archive

    ## Define a reclassification rule
    cat_list=df_weight['cat'].tolist()
    weight_list=df_weight['weight_after_log'].tolist()
    rule=""
    for i, cat in enumerate(cat_list):
        rule+=str(cat)
        rule+="="
        rule+=str(int(round(weight_list[i]*1000000000,0)))  #reclass rule of r.reclass requier INTEGER but random forest prediction could be very low values.
        rule+="\n"
    rule+="*"
    rule+="="
    rule+="NULL"

    ## Create a temporary 'weight_reclass_rules.csv' file for r.reclass
    outputcsv=os.path.join(outputdirectory_grid,"weight_reclass_rules.csv")
    TMP_CSV.append(outputcsv)
    f = open(outputcsv, 'w')
    f.write(rule)
    f.close()

    ## Reclass segments raster layer
    gscript.run_command('g.region', raster='clumped_grid')
    gscript.run_command('r.reclass', quiet=True, overwrite=True, input="clumped_grid", output="weight_int", rules=outputcsv)
    gscript.run_command('r.mapcalc', expression="weight_float=float(weight_int)/float(1000000000)", quiet=True, overwrite=True) #Get back to the original 'float' prediction of population density of random forest
    TMP_MAPS.append("weight_int")
    TMP_MAPS.append("weight_float")

    ## Force weight to zero if no built-up pixel in the grid
    if built_up =='':
        gscript.run_command('r.mapcalc',expression="%s=weight_float"%weigthing_layer_name, quiet=True, overwrite=True)
    else:
        gscript.run_command('g.region', raster='clumped_grid')
        gscript.run_command('r.resamp.stats', quiet=True, overwrite=True, input='class_%s'%built_up, output='sum_lc_%s'%built_up, method='sum')
        gscript.run_command('r.mapcalc',expression="%s=if(sum_lc_%s!=0,weight_float,0)"%(weigthing_layer_name,built_up), quiet=True, overwrite=True)
        TMP_MAPS.append('sum_lc_%s'%built_up)

    # -------------------------------------------------------------------------
    # Feature importances
    # -------------------------------------------------------------------------

    importances = regressor.feature_importances_  #Save feature importances from the model
    indices = np.argsort(importances)[::-1]
    x_axis = importances[indices][::-1]
    idx = indices[::-1]
    y_axis = range(x.shape[1])
    plt.figure(figsize=(5, (len(y_axis)+1)*0.23))  #Set the size of the plot according to the number of features
    plt.scatter(x_axis,y_axis)
    Labels = []
    for i in range(x.shape[1]):
        Labels.append(x_grid.columns[idx[i]])
    Labels=labels_from_csv(Labels)  #Change the labels of the feature according to 'lc_classes_list' and 'lu_classes_list'
    plt.yticks(y_axis, Labels)
    plt.ylim([-1,len(y_axis)])  #Ajust ylim
    plt.xlim([-0.04,max(x_axis)+0.04]) #Ajust xlim
    plt.title("Feature importances")
    if not os.path.exists(os.path.split(plot)[0]):  #Create folder where to save the plot if not exists
        os.makedirs(os.path.split(plot)[0])
    plt.savefig(plot+'.png', bbox_inches='tight', dpi=400)  # Export in .png file (image)
    plt.savefig(plot+'.svg', bbox_inches='tight', dpi=400)  # Export in .svg file (vectorial)
    message='Final Random Forest model run - internal Out-of-bag score (OOB) : %0.3f'%regressor.oob_score_
    log_text+=message+'\n'
    print message

def main():
    global TMP_MAPS, TMP_CSV, vector, min_fimportance, param_grid, kfold, gridded_vector, Land_cover, Land_use, distance_to, tile_size, n_jobs, id, population, built_up, output, plot, log_file, log_text, log_text_extend, nsres, ewres, lc_classes_list, lu_classes_list, lc_class_name, lu_class_name
    TMP_MAPS = []
    TMP_CSV = []
    start_time=time.ctime()
    log_text=""
    log_text_extend=""
    # user's values
    vector = options['vector']
    Land_cover = options['land_cover']
    Land_use = options['land_use'] if options['land_use'] else ""
    distance_to = options['distance_to'] if options['distance_to'] else ""
    tile_size = options['tile_size']
    id = options['id']
    population = options['population']
    built_up = options['built_up_category'] if options['built_up_category'] else ""
    output_weighting_layer = options['output']
    plot = options['plot']
    log_file = options['log_file'] if options['log_file'] else ""
    lc_list = options['lc_list'].split(",") if options['lc_list'] else ""
    lu_list = options['lu_list'].split(",") if options['lu_list'] else ""
    lc_class_name = options['lc_class_name'] if options['lc_class_name'] else ""
    lu_class_name = options['lu_class_name'] if options['lu_class_name'] else ""
    distance_to = options['distance_to'] if options['distance_to'] else ""
    min_fimportance = 0.00 if flags['a'] else 0.005   # Default value = 0.005 meaning covariates with less than 0.5% of importance will be removed. If flag active, then all covariates will be kept
    kfold = int(options['kfold']) if options['kfold'] else 5  # Default value is 5-fold cross validation
    if options['param_grid']:
        try:
            literal_eval(options['param_grid'])
        except:
            gscript.fatal(_("The syntax of the Python dictionary with model parameter is not as expected. Please refer to the manual"))
    param_grid = literal_eval(options['param_grid']) if options['param_grid'] else {'oob_score': [True],'bootstrap': [True],
                                                                                    'max_features': ['sqrt',0.1,0.2,0.3,0.4,0.5],
                                                                                    'n_estimators': [50, 150, 300, 500, 700]}
    n_jobs = int(options['n_jobs'])

    # vector exists?
    result = gscript.find_file(vector, element='vector')
    if len(result['name']) == 0:
        gscript.fatal(_("Input vector <%s> not found") % vector)

    # Land_cover exists?
    result = gscript.find_file(Land_cover, element='cell')
    if len(result['name']) == 0:
        gscript.fatal(_("Input Land Cover <%s> not found") % Land_cover)

    # id column exists ?
    if id not in gscript.vector_columns(vector).keys():
        gscript.fatal(_("<%s> column does not exist for vector %s") % (id, vector))

    # is id column numeric?
    coltype = gscript.vector_columns(vector)[id]['type']
    if coltype not in ('INTEGER'):
        gscript.fatal(_("<%s> column must be Integer")% (id))

    # population column exists ?
    if population not in gscript.vector_columns(vector).keys():
        gscript.fatal(_("<%s> column does not exist for vector %s") % (population, vector))

    # is population column numeric?
    coltype = gscript.vector_columns(vector)[population]['type']
    if coltype not in ('INTEGER', 'DOUBLE PRECISION'):
        gscript.fatal(_("<%s> column must be numeric")% (population))

    # does population column contain values <=0 of NULL ?
    for x in gscript.parse_command('v.db.select', map=vector, columns=population, null_value='0', flags='c'):
        if float(x) <= 0:
            gscript.fatal(_("Response values contained in <%s> column can not be smaller than 1 or have NULL values. Check manual page for more informations")% (population))

    # is tile_size diffrent from null?
    if (int(tile_size) <= gscript.raster_info(Land_cover).nsres):
        gscript.fatal(_("Invalid Tile size, it must be greater than Land Cover's resolution "))

    # Land_use exists?
    if (Land_use !=''):
        result = gscript.find_file(Land_use, element='cell')
        if (len(result['name']) == 0):
            gscript.fatal(_("Input raster <%s> not found") % Land_use)

    # distance_to exists?
    if (distance_to !=''):
        result = gscript.find_file(distance_to, element='cell')
        if (len(result['name']) == 0):
            gscript.fatal(_("Input distance_to <%s> not found") % distance_to)

    # list of categories exists in the corresponding raster ?
    if (lc_list != ""):
        category_list_check(lc_list, Land_cover)
    if (lu_list != "" ):
        category_list_check(lu_list, Land_use)

    # file with class name exists ?
    if lc_class_name != "":
        if os.path.isfile(lc_class_name) is False :
            gscript.fatal(_("Csv file containing class names for <%s> doesn't exists") % Land_cover)
    if lu_class_name != "":
        if os.path.isfile(lu_class_name) is False :
            gscript.fatal(_("Csv file containing class names for <%s> doesn't exists") % Land_use)

    # Check if 'oob_score' parameter in the dictionnary for grid search is well True
    if 'oob_score' not in param_grid.keys():
        param_grid['oob_score']=[True]
    elif param_grid['oob_score']!=[True]:
        param_grid['oob_score']=[True]
   
    # valid n_jobs?
    if(n_jobs >= multiprocessing.cpu_count()):
        gscript.fatal(_("Requested number of jobs is > or = to available ressources. Try to reduce to at maximum <%s> jobs")%(int(multiprocessing.cpu_count())-1))

    # Is kfold valid ?
    maxfold=int(gscript.parse_command('v.db.univar', flags='g', map=vector, column='cat')['n'])   ## Corespond to leave-one-out cross-validation 
    if(kfold > maxfold):
        gscript.fatal(_("<kfold> parameter must be lower than %s (number of administratives area)"%maxfold))
    if(kfold < 2):
        gscript.fatal(_("<kfold> parameter must be higher than 2"))
                      
    # Check if i.segment.stats is well installed
    if not gscript.find_program('i.segment.stats', '--help'):
        message = _("You first need to install the addon i.segment.stats.\n")
        message += _(" You can install the addon with 'g.extension i.segment.stats'")
        gscript.fatal(message)

    ## Create temporary directory(-ies) for output
    create_tempdirs()

    ## Creating a empty grid raster: each grid has a size corresponding to the "tile_size" parameter
    create_clumped_grid(tile_size)

    ## Creating a raster corresponding to administratives zones - with resolution corresponding to the 'landcover' raster
    admin_boundaries(vector.split("@")[0], id)

    # Data preparation : extract list of classes from the Land Cover
    if (lc_list == ""):
        lc_classes_list = Data_prep(Land_cover)[2]  #Get a sorted list with values of category in this raster
    else:
        lc_classes_list = lc_list
        lc_classes_list.sort(key=float)  #Make sur the list provided by the user is well sorted.
    message="Classes of raster '"+str(Land_cover)+"' used: "+",".join(lc_classes_list)
    log_text+=message+'\n'
    print message

    # Data preparation : extract list of classes from the land use
    if(Land_use != '' ):
        if (lu_list == ""):
            lu_classes_list = Data_prep(Land_use.split("@")[0])[2]  #Get a sorted list with values of category in this raster
        else:
            lu_classes_list = lu_list
            lu_classes_list.sort(key=float)  #Make sur the list provided by the user is well sorted.
        message="Classes of raster '"+str(Land_use)+"' used: "+",".join(lu_classes_list)
        log_text+=message+'\n'
        print message

    ## Compute proportion of each class of categorical raster (parallel processing).
    #for landcover
    gscript.run_command('g.region', raster=Land_cover.split("@")[0])  #Set the region to match the extend of the raster
    p=Pool(n_jobs) #Create a 'pool' of processes and launch them using 'map' function
    func=partial(proportion_class,Land_cover.split("@")[0]) # Set fixed argument of the function
    output=p.map(func,lc_classes_list) # Launch the processes for as many items in the list (if function with a return, the returned results are ordered thanks to 'map' function)
    p.close()
    p.join()
    temp_rasterlist,temp_csvlist_1,temp_csvlist_2=zip(*output)
    [TMP_MAPS.append(x) for x in temp_rasterlist]  # Append the name of binary rasters to the list of temporary maps
    [TMP_CSV.append(x) for x in temp_csvlist_1]  # Append the paths to .csv files to the list of temporary .csv
    [TMP_CSV.append(x) for x in temp_csvlist_2]  # Append the paths to .csv files to the list of temporary .csv

    #for landuse
    if(Land_use != '' ):
        gscript.run_command('g.region', raster=Land_use.split("@")[0])  #Set the region to match the extend of the raster
        p=Pool(n_jobs) #Create a 'pool' of processes and launch them using 'map' function
        func=partial(proportion_class,Land_use.split("@")[0]) # Set fixed argument of the function
        output=p.map(func,lu_classes_list) # Launch the processes for as many items in the list (if function with a return, the returned results are ordered thanks to 'map' function)
        p.close()
        p.join()
        temp_rasterlist,temp_csvlist_1,temp_csvlist_2=zip(*output)
        [TMP_MAPS.append(x) for x in temp_rasterlist]  # Append the name of binary rasters to the list of temporary maps
        [TMP_CSV.append(x) for x in temp_csvlist_1]  # Append the paths to .csv files to the list of temporary .csv
        [TMP_CSV.append(x) for x in temp_csvlist_2]  # Append the paths to .csv files to the list of temporary .csv

    ## Compute mean distance to places of interest
    if(distance_to !=''):
        # For grids
        grid_stat_output=os.path.join(outputdirectory_grid,"mean_dist.csv")
        TMP_CSV.append(grid_stat_output)
        gscript.run_command('i.segment.stats', quiet=True, overwrite=True, map='clumped_grid', area_measures="", rasters=distance_to.split("@")[0], raster_statistics='mean', csvfile=grid_stat_output, separator='comma')
        # For administrative zones
        admin_stat_output=os.path.join(outputdirectory_admin,"mean_dist.csv")
        TMP_CSV.append(admin_stat_output)
        gscript.run_command('i.segment.stats', quiet=True, overwrite=True, map='gridded_admin_units', area_measures="", rasters=distance_to.split("@")[0], raster_statistics='mean', csvfile=admin_stat_output, separator='comma')
        # Save log
        log_text+='Distance raster used : '+str(distance_to)+'\n\n'

    ## Join .csv files of statistics
    for directory in [outputdirectory_grid, outputdirectory_admin]:
        allstatfile=os.path.join(directory,"all_stats.csv")
        pattern_A="LC_*_prop.csv"   #Add all csv with proportions of Land cover classes
        pattern_B=""
        pattern_C=""
        if(Land_use != '' ):
            pattern_B="LU_*_prop.csv"  #Add all csv with proportions of Land use classes
        if distance_to !='':
            pattern_C="mean_*.csv"     #Add mean distance to amenity if the option was used
        list_paths=ordered_list_of_path(directory,pattern_A,pattern_B,pattern_C)  #Get ordered list of path
        join_multiplecsv(list_paths,allstatfile,separator=",",join='inner', fillempty='NULL', overwrite=True)

    ## Random Forest
    RandomForest(output_weighting_layer,vector.split("@")[0],id)

    ## Export the log file
    end_time=time.ctime()
    logging=open(log_file+'.txt', 'w')
    logging.write('Log file of r.population.density\n')
    logging.write('Run started on '+str(start_time)+' and finished on '+str(end_time)+'\n')
    logging.write('Selected spatial resolution for weighting layer : '+tile_size+' meters\n')
    logging.write('Administrative layer used : '+vector+'\n')
    logging.write(log_text)
    if flags['f'] :
        logging.write("\n")
        logging.write(log_text_extend)
    logging.close()

# exécution
if __name__ == "__main__":
    options, flags = gscript.parser()
    atexit.register(cleanup)
    main()

