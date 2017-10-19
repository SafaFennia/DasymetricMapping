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
#*             
#*
#*             This program is free software under the
#*             GNU General Public License (>=v2).
#*             Read the file COPYING that comes with GRASS
#*             for details.
#*
#****************************************************************
 
#%module
#% description: Creates a weighting layer with Random Forest for population redistribution
#% keywords: Land Cover, Density, Population, Dasymetry, Random Forest
#%end
#%option G_OPT_V_INPUT
#% key: vector
#% description: Input file: Vector where the attribute table contains the ID,the census data and the area of each administrative unit 
#% required : yes
#%end
#%option G_OPT_R_INPUT
#% key: land_cover
#% type: string
#% description: Input file: Land Cover
#% required : yes
#%end
#%option G_OPT_R_INPUT
#% key: raster
#% type: string
#% description: Input file: raster( Land use, morphological areas...)
#% required : no
#%end
#%option G_OPT_R_INPUT
#% key: distance_to
#% type: string
#% description: Input raster: distance to zones of interest
#% required : no
#%end
#%option
#% key: tile_size
#% key_desc: value
#% type: Integer
#% description: Tile_size
#% required : yes
#%end
#%option
#% key: id
#% type: string
#% description: Identifier column for administrative units
#% required : yes
#%end
#%option
#% key: population
#% type: string
#% description: Census data column
#% required : yes
#%end
#%option
#% key: area
#% type: string
#% description: Area column
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
#% key: png
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
#% label: list of additional raster categories to be used
#% description: Format: 1 2 3 thru 7 *
#% required: no
#%end
#%option
#% key: n_jobs
#% type: integer
#% description: Size of the thread-pool for the parallel process
#% required : yes
#%end



 
import os
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

from joblib import Parallel, delayed


from multiprocessing import Process, Manager, Value, Lock
import multiprocessing
import logging

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

	

def cleanup():
    gscript.run_command('g.remove', type='raster', name=','.join(TMP_MAPS), flags='f')	
	

	
def Data_prep(Land_cover):
	'''
	Function that extracts informations from the Land Cover : resolution and classes
	'''
	
	info = gscript.raster_info(Land_cover)
	nsres=info.nsres
	ewres=info.ewres
	L = []
	L=[cl for cl in gscript.parse_command('r.category',map= Land_cover)]
	return nsres, ewres, L

	

	
def proportion_class_grid(rasterLayer, vectorLayer, tile_size,nsres,ewres, cl, opt=""):
	'''
	Function calculating the proportion of each class within a grid of ('tile_size' x 'tile_size')
	'''

	gscript.run_command('g.region', raster=rasterLayer, nsres=nsres, ewres=ewres)
	#add a new column
	gscript.run_command('v.db.addcolumn', map_=vectorLayer, column= "proportion_"+str(cl)+opt+" double precision")
	#calculate the class's proportion
	building = 'binary_class_'+str(cl)+opt
	TMP_MAPS.append(building)
	gscript.run_command('r.mapcalc', expression=building+'=if('+rasterLayer+'=='+str(cl)+',1,0)',overwrite=True)
	proportion='proportion_binary_class_'+str(cl)+opt
	TMP_MAPS.append(proportion)
	somme = 'somme_binary_classs'+str(cl)+opt
	TMP_MAPS.append(somme)
	gscript.run_command('g.region', raster= rasterLayer, res=tile_size)
	gscript.run_command('r.resamp.stats',  input=building, output=somme, method='sum',overwrite=True)#Somme
	gscript.run_command('r.mapcalc', expression= proportion+'= ((('+ somme+')*('+str(ewres*nsres)+')  / ('+str(int(tile_size)**2)+'))*100) ', overwrite=True)
	#filling the attribute table whith this proportion
	gscript.run_command('v.what.rast', map_=vectorLayer, type='centroid', raster=proportion, column= "proportion_"+str(cl)+opt)

		
		

def admin_boundaries(vector, id):
	'''
	Function convecting the vecotor to raster then raster to vector: boundaries will have a staircase appearence 
	so that each tile of the gridded vector will be contained in only one administrative unit
	'''
	
	gscript.run_command('g.region', raster=Land_cover.split("@")[0], res=tile_size)
	gscript.run_command('v.to.rast', input= vector, type='area', output=vector+'_rast', use='cat', overwrite=True)
	gscript.run_command('r.to.vect', input=vector+'_rast', output= vector+'_rastTovect', type='area', column=id, flags='v',overwrite=True)
	gscript.run_command('v.db.join', map=vector+'_rastTovect', column='cat',other_table=vector, other_column='cat')

	

def proportion_class_admin(vector, cl, opt=""):
	'''
	Function calculating classes' proportions within each administrative unit
	'''

	gscript.run_command('v.rast.stats', map=vector+'_rastTovect',raster='proportion_binary_class_'+str(cl)+opt, column_prefix='proportion_'+str(cl)+opt, method='average', flags='c')


		
	
def RandomForest(vector,id):
	'''
	Function that creates a random forest model trained at the administrative units level to generate gridded prediction
	covariates are proportion of each Land Cover's class (opt: with proportion of each morpho zone)
	'''
	
	# -------------------------------------------------------------------------
    # Data preparation for administrative units
    # -------------------------------------------------------------------------
	
	#add a new column
	gscript.run_command('v.db.addcolumn', map_=vector+'_rastTovect', column= "log_population_density double precision")
	#filling the column : pb with v.db.update and log.
	#solution: filling column through a csv file using pandas library
	#export the attribute table to a csv file
	csv_table = os.path.join(tempfile.gettempdir(),vector+'_rastTovect.csv')
	gscript.run_command('db.out.ogr', input=vector+'_rastTovect', output=csv_table , format='CSV', overwrite=True)
	#reading the csv file as dataframe
	attr_table = pd.read_csv(csv_table)
	#filling the log population density column
	log = attr_table['log_population_density']
	pop = attr_table[population]
	admin_area = attr_table[area]
	for i in range(len(log.values)):
		log.values[i]=ma.log(pop.values[i]/admin_area.values[i])
	#export the dataframe to the csv
	attr_table.to_csv(path_or_buf=csv_table)

	# -------------------------------------------------------------------------
    # Data preparation for grids
    # -------------------------------------------------------------------------
	
	## Create a temporary csv file
	csv_table_grid = os.path.join(tempfile.gettempdir(),"grid_vector.csv")
	#export the attribute table to a csv file
	gscript.run_command('db.out.ogr', input="grid_vect", output=csv_table_grid , format='CSV', overwrite=True)


	

	# -------------------------------------------------------------------------
    # Creating RF model
    # -------------------------------------------------------------------------
	
	df_admin = pd.read_csv(csv_table)
	df_grid = pd.read_csv(csv_table_grid)
	min_class = min(lc_classes_list)
	max_class = max(lc_classes_list)

	df_admin = df_admin.reindex_axis(sorted(df_admin.columns), axis=1)
	df_grid = df_grid.reindex_axis(sorted(df_grid.columns), axis=1)
	## changing null values to zero
	features = df_grid.columns[:]
	for i in features:
		for j in range(len(df_grid)):
			if(type(df_grid[i].values[j]) != unicode):
				if(ma.isnan(float(df_grid[i].values[j]))):
					df_grid[i].values[j] = 0
	y = df_admin['log_population_density']
	if (morpho_zones == ''):			
		x = df_admin.loc[:,'proportion_'+str(min_class)+'_average':'proportion_'+str(max_class)+'_average']
		x_grid = df_grid.loc[:,'proportion_'+str(min_class) :'proportion_'+str(max_class)]
		if(distance_to != ''): 
			x_distance = df_admin['distance_average']
			x_grid_distance = df_grid['distance']
			x = pd.concat((x,x_distance), axis=1)
			x_grid = pd.concat((x_grid,x_grid_distance), axis=1)
	else:
		min_morpho = min(morpho_classes_list)
		max_morpho = max(morpho_classes_list)
		min_category = min(min_morpho,min_class)
		max_category = max(max_morpho,max_class)
		c = '_morpho' if (max_category == max_morpho) else ""


		x = df_admin.loc[:,'proportion_'+str(min_category):'proportion_'+str(max_category)+c+"_average"]
		x_grid = df_grid.loc[:,'proportion_'+str(min_category):'proportion_'+str(max_category)+c]
		if(distance_to != ''): 
			x_distance = df_admin['distance_average']
			x_grid_distance = df_grid['distance']
			x = pd.concat((x,x_distance), axis=1)
			x_grid = pd.concat((x_grid,x_grid_distance), axis=1)
			
	
	
	regressor = RandomForestRegressor(n_estimators = 200, oob_score = True)
	regressor.fit(x, y)

	prediction = regressor.predict(x_grid)

	df1 = df_grid['cat']
	df2 = pd.DataFrame(prediction, columns=['log'])
	df_weight = pd.concat((df1,df2), axis=1)
	col = df_weight.apply ( lambda row : np.exp(row["log"]), axis = 1 ) 
	df_weight ["weight_after_log"] = col
	weightcsv=os.path.join(tempfile.gettempdir(),"weight.csv")
	df_weight.to_csv(path_or_buf=weightcsv)
	
	gscript.run_command('g.region', raster=Land_cover.split("@")[0], res=tile_size)
	# create a csvt file containing columns types
	f = open(weightcsv+"t", 'w')
	header_string = '"Integer", "Integer","Real","Real"'
	f.write(header_string)
	f.close()


	
	#importing the csv to the database
	gscript.run_command('db.in.ogr', input=weightcsv, output='weight_table', overwrite=True)
	# join the two tables
	gscript.run_command('v.db.join', map='grid_vect', column='cat', other_table='weight_table', other_column='cat_')
	#converting the vector layer to a raster one containing the weight generated by RF
	gscript.run_command('v.to.rast', input='grid_vect', type='area', output='RFweight',use='attr', attribute_column='weight_after_log', overwrite=True)
	gscript.run_command('g.region', raster=Land_cover.split("@")[0], res=tile_size)
	#assigning 0 value to pixels that are outside the study area
	gscript.run_command('r.mapcalc', expression="RFweight_Bis = if( "+vector+"_rast !=0, RFweight,0)",overwrite=True)



	if built_up =='':
		gscript.run_command('r.mapcalc',expression=output+" = RFweight_Bis", overwrite=True)
	else:
		gscript.run_command('r.mapcalc',expression=output+" = if( proportion_binary_class_"+str(built_up)+" !=0,RFweight_Bis,0)", overwrite=True)
		

	

	# -------------------------------------------------------------------------
    # Feature importances
    # -------------------------------------------------------------------------
	
	importances = regressor.feature_importances_
	indices = np.argsort(importances)[::-1]
	x_axis = importances[indices][::-1]
	idx = indices[::-1]
	y_axis = range(x.shape[1])
	plt.scatter(x_axis,y_axis)
	Labels = []
	for i in range(x.shape[1]):
		Labels.append(x_grid.columns[idx[i]])
	plt.yticks(y_axis, Labels)
	plt.title("Feature importances")
	plt.savefig(png+'.png', bbox_inches='tight')
	
	
	
	print('oob_score = ',regressor.oob_score_)
	


def main():
	global TMP_MAPS , vector, Land_cover, morpho_zones,distance_to, tile_size, id, population, area,built_up, output, png, nsres,ewres, lc_classes_list, morpho_classes_list
	TMP_MAPS = []
	# user's values
	vector = options['vector']
	Land_cover = options['land_cover']
	morpho_zones = options['raster'] if options['raster'] else ""
	distance_to = options['distance_to'] if options['distance_to'] else ""
	tile_size = options['tile_size']
	id = options['id']
	population = options['population']
	area = options['area']
	built_up = options['built_up_category'] if options['built_up_category'] else ""
	output = options['output']
	png = options['png'] 
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

    # is  id column numeric?
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
	
	# area column exists ?
	if area not in gscript.vector_columns(vector).keys():
		gscript.fatal(_("<%s> column does not exist for vector %s") % (area, vector))
	
	# is  area column numeric?
	coltype = gscript.vector_columns(vector)[area]['type']
	if coltype not in ('INTEGER', 'DOUBLE PRECISION'):
		gscript.fatal(_("<%s> column must be numeric")% (area))	

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
	# data preparation : extract classes and resolution from the Land Cover 
	Data = Data_prep(Land_cover)
	nsres = Data[0] ## north-south resolution
	ewres = Data[1] ## east-west resolution
	if (lc_list == ""):
		lc_classes_list = Data[2]
	else:
		lc_classes_list = lc_list
	
	
	

	
	## creating a gridded vector: each grid has a size of "tile_size"
	gscript.run_command('g.region', raster= Land_cover.split("@")[0], res=tile_size)
	gscript.mapcalc("empty_grid=rand(0 ,999999999 )", overwrite=True,seed='auto' ) #creating a raster with random values
	gscript.run_command('r.clump',input='empty_grid',output='empty_grid_Clump',overwrite=True) #assigning a unique value to each grid  
	gscript.run_command('r.to.vect',input='empty_grid_Clump',output='grid_vect', type='area', overwrite=True) #raster to vector
	
	
	## calculating classes' proportions within each grid and administrative unit
	admin_boundaries(vector.split("@")[0], id)
	grid_lc = Parallel(n_jobs=n_jobs,backend="threading")(delayed(proportion_class_grid, check_pickle=False)(Land_cover.split("@")[0], 'grid_vect', tile_size, nsres, ewres, cl ,"",) for cl in lc_classes_list)
	admin_lc = Parallel(n_jobs=n_jobs,backend="threading")(delayed(proportion_class_admin, check_pickle=False)(vector.split("@")[0], cl ,"",) for cl in lc_classes_list)
	if(morpho_zones != '' ):
			if (morpho_list == ""):
				morpho_classes_list = Data_prep(morpho_zones.split("@")[0])[2]
			else:
				morpho_classes_list = morpho_list
			grid_morpho = Parallel(n_jobs=n_jobs,backend="threading")(delayed(proportion_class_grid, check_pickle=False)(morpho_zones.split("@")[0], 'grid_vect', tile_size, nsres, ewres,cl, "_morpho",) for cl in morpho_classes_list)
			admin_morpho = Parallel(n_jobs=n_jobs,backend="threading")(delayed(proportion_class_admin, check_pickle=False)(vector.split("@")[0], cl, "_morpho",) for cl in morpho_classes_list)
	
	## adding distance to places of interest data to the attribute table of the gridded vector and calculate its mean for each administrative unit
	if(distance_to !=''):
		gscript.run_command('v.what.rast', map='grid_vect', type='centroid', raster=distance_to, column='distance')
		gscript.run_command('v.rast.stats', map=vector.split("@")[0]+'_rastTovect', raster=distance_to, column_prefix='distance', method='average', flags='c')
	
	## Random Forest
	RandomForest(vector.split("@")[0],id)
	
	
	
	





 
# exécution
if __name__ == "__main__":
	options, flags = gscript.parser()
	atexit.register(cleanup)
	main()

