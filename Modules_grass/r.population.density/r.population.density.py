#!/usr/bin/env python
# encoding: utf-8
#****************************************************************
#*
#* MODULE:     r.population.density
#*
#* AUTHOR(S):  Safa Fennia
#*
#* PURPOSE:    Create a population density map
#*
#* COPYRIGHT:  (C) 2017 Safa Fennia
#*             ANAGEO, Université libre de Bruxelles, Belgique
#*
#*
#*             First Version: 2017/07/18
#*             Second Version: 2017/11/05
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
#% key: raster
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
#% description: Name for output plot
#% required: yes
#% guisection: Output
#%end
#%option
#% key: lc_list
#% type: string
#% label: Land Cover's categories to be used
#% description: Format: 1 2 3 thru 7 *
#% required: no
#%end
#%option
#% key: raster_list
#% type: string
#% label: List of additional raster categories to be used
#% description: Format: 1 2 3 thru 7 *
#% required: no
#%end
#%option
#% key: n_jobs
#% type: integer
#% description: Number of cores to be used for the parallel process
#% required : yes
#%end




import os, sys, glob, time
import csv
import atexit
## Import GRASS GIS Python Scripting Library
import grass.script as gscript

## Import library for temporary files creation
import tempfile

## Import Numpy library
import numpy as np

## import math library
import math as ma

## import pyplot library
import matplotlib
matplotlib.use('Agg') #use a non-interactive backend: prevent the figure from popping up
import matplotlib.pyplot as plt

import multiprocessing
from multiprocessing import Process, Manager, Value, Lock
import logging

## Import Joblib library (multiprocessing)
try:
    from joblib import Parallel, delayed
except:
    gscript.fatal("Joblib is not installed ")

## Import Pandas library (View and manipulaiton of tables)
try:
    import pandas as pd
except:
    gscript.fatal("Pandas is not installed ")

## Import Random Forest library
try:
    from sklearn.ensemble import RandomForestRegressor
except:
    gscript.fatal("Scikit learn 0.18 or newer is not installed")

##TODO: Add check for presence of negative of null values in the population column of the administrative unit vector layer. If presence, a error message should appeared because ma.log() would crash anyway

def cleanup():
    gscript.run_command('g.remove', quiet=True, type='raster', name=','.join(TMP_MAPS), flags='fb')   ##TODO Add removing of temporary csv files


def create_tempdirs():
    # Temporary directory for administrative units statistics
    global outputdirectory_admin
    outputdirectory_admin=os.path.join(tempfile.gettempdir(),"admin_stats")
    if not os.path.exists(outputdirectory_admin):
        os.makedirs(outputdirectory_admin)
    # Temporary directory for grids statistics
    global outputdirectory_grid
    outputdirectory_grid=os.path.join(tempfile.gettempdir(),"grid_stats")
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


def admin_boundaries(vector, id):
    '''
    Function convecting the vecotor to raster then raster to vector: boundaries will have a staircase appearence
    so that each tile of the gridded vector will be contained in only one administrative unit
    '''
    gscript.run_command('g.region', raster='clumped_grid')
    gscript.run_command('v.to.rast', quiet=True, input=vector, type='area', output='gridded_admin_units', use='attr', attribute_column=id, overwrite=True)
    gscript.run_command('r.to.vect', quiet=True, input='gridded_admin_units', output=vector.split("@")[0]+'_'+str(tile_size)+'m_gridded', type='area', column=id, flags='v',overwrite=True)
    gscript.run_command('v.db.join', map_=vector.split("@")[0]+'_'+str(tile_size)+'m_gridded', column='cat', other_table=vector, other_column=id, subset_columns=population) #join the population count
    print "Create raster of administrative units with spatial resolution of "+str(tile_size)+" meters"
    TMP_MAPS.append("gridded_admin_units")


def area_gridded_admin():
    '''
    Function calculating classes' proportions within each administrative unit
    '''
    #compute sum of pixels of the current class
    admin_area_output=os.path.join(outputdirectory_admin,"area.csv")
    gscript.run_command('i.segment.stats', flags="r", quiet=True, overwrite=True, map='gridded_admin_units', area_measures='area', csvfile=admin_area_output, separator='comma')
    print "Computing area of (gridded) administrative units"

    # Create a list with area of each administrative zones
    global area_list
    area_list=[]
    tmp_list=[]
    f=open(admin_area_output)
    f.next() #Pass the first line containing column headers
    for row in f:
        area_list.append(row.split("\n")[0].split(",")[1]) #For each line, save the second column in the list named 'area_list'


def proportion_class_admin(cl, opt=""):
    '''
    Function calculating classes' proportions within each administrative unit
    '''
    #compute sum of pixels of the current class
    if opt=="":
        admin_stat_csv=os.path.join(outputdirectory_admin,"lc_sum_"+cl+".csv")
    else:
        admin_stat_csv=os.path.join(outputdirectory_admin,opt+"_sum_"+cl+".csv")
    gscript.run_command('i.segment.stats', quiet=True, overwrite=True, map='gridded_admin_units', area_measures="", rasters="class_"+cl+opt, raster_statistics='sum', csvfile=admin_stat_csv, separator='comma')
    #Transform the value in the .csv into a proportion
    nsres, ewres = Data_prep("class_"+cl+opt)[0:2] #Get the north-south and east-west resolution of the current raster
    compute_proportion_csv(admin_stat_csv, type_raster="admin") #Create a new csv containing the proportion


def create_clumped_grid(tile_size):
    gscript.run_command('g.region', raster=Land_cover.split("@")[0], res=tile_size)
    gscript.run_command('r.mask', raster=Land_cover.split("@")[0])
    gscript.mapcalc("empty_grid=rand(0 ,999999999)", overwrite=True, seed='auto') #Creating a raster with random values
    gscript.run_command('r.clump', quiet=True, input='empty_grid',output='clumped_grid',overwrite=True) #Assigning a unique value to each grid
    gscript.run_command('r.mask', flags='r')
    TMP_MAPS.append("empty_grid")
    TMP_MAPS.append("clumped_grid")
    print "Create a grid raster with spatial resolution of "+str(tile_size)+" meters"


def create_binary_raster(rasterLayer, cl, opt=""):
    #Create a binary raster for the current class
    binary_class = "class_"+cl+opt
    gscript.run_command('r.mapcalc', expression=binary_class+'=if('+rasterLayer+'=='+str(cl)+',1,0)',overwrite=True)
    gscript.run_command('r.null', map=binary_class, null='0')
    TMP_MAPS.append(binary_class)


def compute_proportion_csv(infile, type_raster="grid"):
    fin = open(infile) #Open the infile .csv
    # Set the path to the outputfile and open it
    head, tail = os.path.split(infile)
    root, ext = os.path.splitext(tail)
    outfile=os.path.join(head,"prop_"+root+ext)
    fout = open(outfile, 'w') #Create and open a new .csv to be written

    # Count number of row in the csv file
    nline=0
    for row in fin:
        nline+=1

    # Loop on each row and column of the .csv file and compute proportion
    fin = open(infile)
    for rowid, row in enumerate(fin):
        currentrow=[] #Define an empty list which will be filled with values for the new line to be writen
        if rowid==0:
            currentrow.append(row.split("\n")[0])
        else:
            for colid, col in enumerate(row.split(",")):
                if colid==0:
                    currentrow.append(col)
                else:
                    if type_raster=="admin":  #If proportion computed for the administratives zones
                        proportion_value=((float(col)*(ewres*nsres))/(float(area_list[rowid-1])*float(tile_size)**2))*100
                    else:  #If proportion computed for the regular grids
                        proportion_value=((float(col)*(ewres*nsres))/(float(tile_size)**2))*100
                    currentrow.append('{0:.8f}'.format(proportion_value))  #Round to 8 decimals and add the value in the current row
        fout.write(",".join(currentrow))  #Write the new row
        if rowid < nline-1:
            fout.write("\n")  #Write a return on a new line if not at the last row


def proportion_class_grid(cl, opt=""):   #If used with a landuse raster, 'opt' parameter should be like 'opt="_landuse"'
    '''
    Function calculating the proportion of each class within a grid of ('tile_size' x 'tile_size')
    '''
    #compute sum of pixels of the current class in each grid
    if opt=="":
        grid_stat_csv=os.path.join(outputdirectory_grid,"lc_sum_"+cl+".csv")
    else:
        grid_stat_csv=os.path.join(outputdirectory_grid,opt+"_sum_"+cl+".csv")
    gscript.run_command('i.segment.stats', quiet=True, overwrite=True, map='clumped_grid', area_measures="", rasters="class_"+cl+opt, raster_statistics='sum', csvfile=grid_stat_csv, separator='comma')
    #Transform the value in the .csv into a proportion
    global nsres, ewres
    nsres, ewres = Data_prep("class_"+cl+opt)[0:2] #Get the north-south and east-west resolutions of the current raster
    compute_proportion_csv(grid_stat_csv, type_raster="grid") #Create a new csv containing the proportion



def atoi(text):   #Return integer if text is digit - Used in 'natural_keys' function
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


def join_csv(indir,outfile,pattern_A,pattern_B="",pattern_C=""):
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
    # Count number of row in the first csv file
    nline=0
    for row in open(csvList[0]):
        nline+=1

    # Open all the .csv files and the output file
    files = [open(f) for f in csvList] #Make list of open files
    fout = open(outfile, 'w')

    # Declare a variable counting for number of warning events
    warning_event=0

    for rowid in range(nline):
        current_row=[]
        for f in files:
            if f == files[0]:
                current_row.append((f.readline().strip())) #For the first file, copy all the line
            else:
                second_column_item=f.readline().strip().split(",")[1]
                if second_column_item == "":
                    warning_event+=1
                    print "WARNING: .csv files to be merged do not have to same number of lines"
                else:
                    current_row.append(second_column_item) #Strip removes trailing newline  #Copy the second column of the file if not empty
        fout.write(",".join(current_row))
        if rowid < nline-1:
            fout.write("\n")  #Write a return on a new line if not at the last row)
    time.sleep(0.5) #Sleep the process for 0.5 second to be sure the last line will be written
    fout.close() #Close the outptufile

    if warning_event>=1: #If at least one warning event happend
        sys.exit("Unexpected results seem present in the temporay .csv file. Please check")


def RandomForest(vector,id):
    '''
    Function that creates a random forest model trained at the administrative units level to generate gridded prediction
    covariates are proportion of each Land Cover's class (opt: with proportion of each morpho zone)
    '''

    # -------------------------------------------------------------------------
    # Data preparation for administrative units
    # -------------------------------------------------------------------------
    # Join to each gridded administrative unit the population, the area and all the statistics
    gscript.run_command('db.in.ogr', quiet=True, overwrite=True, input=os.path.join(outputdirectory_admin,"area.csv"), output='area_admin') #Import the temporary .csv containing the area of the gridded administrative unit
    gscript.run_command('db.in.ogr', quiet=True, overwrite=True, input=os.path.join(outputdirectory_admin,"all_stats.csv"), output='stat_admin') #Import the temporary .csv containing the statistics of the gridded administrative unit
    gscript.run_command('v.db.join', quiet=True, map_=vector.split("@")[0]+'_'+str(tile_size)+'m_gridded', column='cat', other_table='area_admin', other_column='cat_', subset_columns='area') # Join the table containing the area
    gscript.run_command('v.db.join', quiet=True, map_=vector.split("@")[0]+'_'+str(tile_size)+'m_gridded', column='cat', other_table='stat_admin', other_column='cat_') # Join the table containing the statistics
    # Compute the log of population density
    gscript.run_command('v.db.addcolumn', quiet=True, map_=vector.split("@")[0]+'_'+str(tile_size)+'m_gridded', column= "log_population_density double precision")  #add a new column which will contain the log of density
    admin_attribute_table=os.path.join(outputdirectory_admin,"admin_attribute.csv") # Define the path to the .csv
    gscript.run_command('db.out.ogr', quiet=True, overwrite=True, input=vector.split("@")[0]+'_'+str(tile_size)+'m_gridded', output=admin_attribute_table , format='CSV') #Export the attribute table in .csv
    attr_table = pd.read_csv(admin_attribute_table) #reading the csv file as dataframe
    # Filling the log population density column
    log = attr_table['log_population_density']
    pop = attr_table[population]
    admin_area = attr_table['area']
    for i in range(len(log.values)):
        log.values[i]=ma.log(pop.values[i]/(admin_area.values[i]*(float(tile_size)**2)))  #Compute the log (ln) of the population density
    attr_table.to_csv(path_or_buf=admin_attribute_table, index=False) #export the dataframe to the csv

    # -------------------------------------------------------------------------
    # Creating RF model
    # -------------------------------------------------------------------------
    df_admin = pd.read_csv(admin_attribute_table)
    csv_table_grid=os.path.join(outputdirectory_grid,"all_stats.csv")
    df_grid = pd.read_csv(csv_table_grid)

    ## Changing null values to zero
    # for df_grid
    features = df_grid.columns[:]
    for i in features:
        df_grid[i].fillna(0, inplace=True)
    # for df_admin
    features = df_admin.columns[:]
    for i in features:
        df_admin[i].fillna(0, inplace=True)

    ## Make a list with name of covariables columns
    list_covar=[]
    for lc in lc_classes_list:
        list_covar.append("class_"+lc+"_sum")
    if (morpho_zones != ''):
        for morpho in morpho_classes_list:
            list_covar.append("class_"+morpho+"morpho_sum")
    if(distance_to != ''):
        list_covar.append(distance_to+"_mean")
    first_covar = list_covar[0]   #Save the name of the first covariate to be used
    last_covar = list_covar[-1]  #Save the name of the last covariate to be used

    ## Saving variable to predict (dependent variable)
    y = df_admin['log_population_density']

    ## Saving covariable for prediction (independent variables)
    x=df_admin[list_covar]  #Get a dataframe with independent variables for administratives units
    x_grid=df_grid[list_covar] #Get a dataframe with independent variables for grids
    #x.to_csv(path_or_buf=os.path.join(outputdirectory_admin,"covar_x.csv"), index=False) #Export in .csv for archive
    #x_grid.to_csv(path_or_buf=os.path.join(outputdirectory_grid,"covar_x_grid.csv"), index=False) #Export in .csv for archive

    # Train the random forest regression model on administratives zones
    regressor = RandomForestRegressor(n_estimators = 200, oob_score = True)
    regressor.fit(x, y)
    # Predict on grids
    prediction = regressor.predict(x_grid)

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
        rule+=str(int(round(weight_list[i]*10000,0))) #Reclasse with random forest prediction of population density by hectares
        rule+="\n"
    rule+="*"
    rule+="="
    rule+="NULL"

    ## Create a temporary 'weight_reclass_rules.csv' file
    outputcsv=os.path.join(outputdirectory_grid,"weight_reclass_rules.csv")
    f = open(outputcsv, 'w')
    f.write(rule)
    f.close()

    ## Reclass segments raster layer to keep only training segments, using the reclas_rule.csv file
    gscript.run_command('g.region', raster='clumped_grid')
    gscript.run_command('r.reclass', quiet=True, overwrite=True, input="clumped_grid", output="weight_int", rules=outputcsv)
    gscript.run_command('r.mapcalc', expression="weight_float=float(weight_int)/float(10000)", quiet=True, overwrite=True)
    TMP_MAPS.append("weight_int")
    TMP_MAPS.append("weight_float")

    ## Erase the temporary 'reclass_rule.csv' file
    #os.remove(outputcsv)

    ## Force weight to zero if no built-up pixel in the grid
    if built_up =='':
        gscript.run_command('r.mapcalc',expression=output+" = weight_float", overwrite=True)
    else:
        gscript.run_command('g.region', raster='clumped_grid')
        gscript.run_command('r.resamp.stats', quiet=True, overwrite=True, input='class_'+str(built_up), output='sum_lc_'+str(built_up), method='sum')
        gscript.run_command('r.mapcalc',expression=output+" = if( sum_lc_"+str(built_up)+" !=0,weight_float,0)", quiet=True, overwrite=True)
        TMP_MAPS.append('sum_lc_'+str(built_up))

    # -------------------------------------------------------------------------
    # Feature importances
    # -------------------------------------------------------------------------

    importances = regressor.feature_importances_
    indices = np.argsort(importances)[::-1]
    x_axis = importances[indices][::-1]
    idx = indices[::-1]
    y_axis = range(x.shape[1])
    plt.figure(figsize=(5, (len(y_axis)+1)*0.23))
    plt.scatter(x_axis,y_axis)
    Labels = []
    for i in range(x.shape[1]):
        Labels.append(x_grid.columns[idx[i]])
    plt.yticks(y_axis, Labels)
    plt.ylim([-1,len(y_axis)])
    plt.xlim([-0.04,max(x_axis)+0.04])
    plt.title("Feature importances")
    plt.savefig(plot+'.png', bbox_inches='tight', dpi=400)

    print('oob_score = '+str(regressor.oob_score_))


def main():
    global TMP_MAPS, vector, Land_cover, morpho_zones, distance_to, tile_size, id, population, built_up, output, plot, nsres, ewres, lc_classes_list, morpho_classes_list
    TMP_MAPS = []

    # user's values
    vector = options['vector']
    Land_cover = options['land_cover']
    morpho_zones = options['raster'] if options['raster'] else ""
    distance_to = options['distance_to'] if options['distance_to'] else ""
    tile_size = options['tile_size']
    id = options['id']
    population = options['population']
    built_up = options['built_up_category'] if options['built_up_category'] else ""
    output = options['output']
    plot = options['plot']
    lc_list = options['lc_list'].split(",") if options['lc_list'] else ""
    morpho_list = options['raster_list'].split(",") if options['raster_list'] else ""
    distance_to = options['distance_to'] if options['distance_to'] else ""
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

    # is  population column numeric?
    coltype = gscript.vector_columns(vector)[population]['type']
    if coltype not in ('INTEGER', 'DOUBLE PRECISION'):
        gscript.fatal(_("<%s> column must be numeric")% (population))

    # is tile_size diffrent from null?
    if (int(tile_size) <= gscript.raster_info(Land_cover).nsres):
        gscript.fatal(_("Invalid Tile size, it must be greater than Land Cover's resolution "))

    # morpho_zones exists?
    if (morpho_zones !=''):
        result = gscript.find_file(morpho_zones, element='cell')
        if (len(result['name']) == 0):
            gscript.fatal(_("Input morpho zones <%s> not found") % morpho_zones)

    # distance_to exists?
    if (distance_to !=''):
        result = gscript.find_file(distance_to, element='cell')
        if (len(result['name']) == 0):
            gscript.fatal(_("Input distance_to <%s> not found") % distance_to)

    # valid n_jobs?
    if(n_jobs > multiprocessing.cpu_count()):
        gscript.fatal(_("Invalid n_jobs <%s>") % n_jobs)

    ## Create temporary directory(-ies) for output
    create_tempdirs()

    ## Creating a empty grid raster: each grid has a size corresponding to the "tile_size" parameter
    create_clumped_grid(tile_size)

    ## Creating a raster corresponding to administratives zones - with resolution corresponding to the 'landcover' raster
    admin_boundaries(vector.split("@")[0], id)

    ## Compute area of administrative zones (raster)
    area_gridded_admin()

    # Data preparation : extract list of classes from the Land Cover
    if (lc_list == ""):
        lc_classes_list = Data_prep(Land_cover)[2]  #Get a sorted list with values of category in this raster
    else:
        lc_classes_list = lc_list
        lc_classes_list.sort(key=float)  #Make sur the list provided by the user is well sorted.
    print "Classes of raster '"+str(Land_cover.split("@")[0])+"' to be used: "+",".join(lc_classes_list)

    # Data preparation : extract list of classes from the land use
    if(morpho_zones != '' ):
        if (morpho_list == ""):
            morpho_classes_list = Data_prep(morpho_zones.split("@")[0])[2]  #Get a sorted list with values of category in this raster
        else:
            morpho_classes_list = morpho_list
            morpho_classes_list.sort(key=float)  #Make sur the list provided by the user is well sorted.
        print "Classes of raster '"+str(morpho_zones.split("@")[0])+"' to be used: "+",".join(morpho_classes_list)

    ## Create binary raster for each class.
    #for landcover
    gscript.run_command('g.region', raster=Land_cover.split("@")[0])  #Set the region to match the extend of the raster
    binary_lc = Parallel(n_jobs=n_jobs,backend="threading")(delayed(create_binary_raster, check_pickle=False)(Land_cover.split("@")[0], cl ,"",) for cl in lc_classes_list)
    #for landuse
    if(morpho_zones != '' ):
        gscript.run_command('g.region', raster=morpho_zones.split("@")[0])  #Set the region to match the extend of the raster
        binary_morpho = Parallel(n_jobs=n_jobs,backend="threading")(delayed(create_binary_raster, check_pickle=False)(morpho_zones.split("@")[0], cl ,"morpho",) for cl in morpho_classes_list)

    ## calculating classes' proportions within each grid and administrative unit
    grid_lc = Parallel(n_jobs=n_jobs,backend="threading")(delayed(proportion_class_grid, check_pickle=False)(cl ,"",) for cl in lc_classes_list)
    admin_lc = Parallel(n_jobs=n_jobs,backend="threading")(delayed(proportion_class_admin, check_pickle=False)(cl ,"",) for cl in lc_classes_list)
    if(morpho_zones != '' ):
            grid_morpho = Parallel(n_jobs=n_jobs,backend="threading")(delayed(proportion_class_grid, check_pickle=False)(cl, "morpho",) for cl in morpho_classes_list)
            admin_morpho = Parallel(n_jobs=n_jobs,backend="threading")(delayed(proportion_class_admin, check_pickle=False)(cl, "morpho",) for cl in morpho_classes_list)

    ## adding distance to places of interest data to the attribute table of the gridded vector and calculate its mean for each administrative unit
    if(distance_to !=''):
        # For grids
        grid_stat_output=os.path.join(outputdirectory_grid,"mean_dist.csv")
        gscript.run_command('i.segment.stats', quiet=True, overwrite=True, map='clumped_grid', area_measures="", rasters=distance_to, raster_statistics='mean', csvfile=grid_stat_output, separator='comma')
        # For administrative zones
        admin_stat_output=os.path.join(outputdirectory_admin,"mean_dist.csv")
        gscript.run_command('i.segment.stats', quiet=True, overwrite=True, map='gridded_admin_units', area_measures="", rasters=distance_to, raster_statistics='mean', csvfile=admin_stat_output, separator='comma')

    ## Join .csv files of statistics
    for directory in [outputdirectory_grid, outputdirectory_admin]:
        allstatfile=os.path.join(directory,"all_stats.csv")
        pattern_A="prop_lc*.csv"   #Add all proportions from Land cover or Land use raster
        pattern_B=""
        pattern_C=""
        if(morpho_zones != '' ):
            pattern_B="prop_morpho*.csv"
        if distance_to !='':
            pattern_C="mean_*.csv"     #Add mean distance to amenity if the option was used
        join_csv(directory,allstatfile,pattern_A,pattern_B,pattern_C)

    ## Random Forest
    RandomForest(vector.split("@")[0],id)


# exécution
if __name__ == "__main__":
    options, flags = gscript.parser()
    atexit.register(cleanup)
    main()

