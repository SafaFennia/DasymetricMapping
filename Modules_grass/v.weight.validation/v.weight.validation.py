
# coding: utf-8


#!/usr/bin/env python
# encoding: utf-8
#****************************************************************
#*
#* MODULE:     v.validation
#*
#* AUTHOR(S):  Safa Fennia
#*
#* PURPOSE:    Calculates the R² and the Root Mean Square Error (RMSE) of aggregated polygons
#*
#* COPYRIGHT:  (C) 2017 Safa Fennia
#*             ANAGEO, Université libre de Bruxelles, Belgique
#*             
#*
#*             First Version: 2017/07/28
#*             
#*
#*             This program is free software under the
#*             GNU General Public License (>=v2).
#*             Read the file COPYING that comes with GRASS
#*             for details.
#*
#****************************************************************
 
#%module
#% description: Calculates the R² and the Root Mean Square Error (RMSE) of aggregated polygons
#% keywords: vector, validation, RMSE, R²
#%end
#%option G_OPT_V_INPUT
#% key: vector
#% description: Input file: Name of input vector map
#% required : yes
#% guisection: Vector
#%end
#%option G_OPT_V_INPUT
#% key: up_vector
#% description: Input file: Name of level-up input vector map 
#% required : yes
#% guisection: Level-up data
#%end
#%option G_OPT_R_INPUT
#% key: weight
#% type: string
#% description: Input file: weighting layer
#% required : yes
#% guisection: Level-up data
#%end
#%option
#% key: id
#% type: string
#% description: Identifier column for administrative units
#% required : yes
#% guisection: Vector
#%end
#%option
#% key: column
#% type: string
#% description: column of observed data
#% required : yes
#% guisection: Vector
#%end
#%option
#% key: up_id
#% type: string
#% description: Identifier column for administrative units 
#% required : yes
#% guisection: Level-up data
#%end
#%option
#% key: up_column
#% type: string
#% description: observed data's column of the level-up vector
#% required : yes
#% guisection: Level-up data
#%end
#%option G_OPT_R_OUTPUT
#% key: output
#% description: Name for output prediction error map
#% required: yes
#% guisection: Output
#%end
#%option G_OPT_F_OUTPUT
#% key: plot
#% description: Name for output plot
#% required: yes
#% guisection: Output
#%end






 
import os
import atexit
## Import GRASS GIS Python Scripting Library
import grass.script as gscript

## Import library for temporary files creation 
import tempfile 

## Import Numpy library
import numpy as np


## Import math library
import math as ma

## Import pyplot library 
import matplotlib
matplotlib.use('Agg') #use a non-interactive backend: prevent the figure from popping up  
import matplotlib.pyplot as plt

## Import sqlite3 library
import sqlite3


## Import Pandas library (View and manipulaiton of tables)
try:
	import pandas as pd
except:
	gscript.fatal("Pandas is not installed ")
	


	


def data_preparation(vector,id):
	'''
	converts the vector to raster then raster to vector: boundaries will have a staircase appearence 
	so that each pisel of the weighting layer will be contained in only one administrative unit
	
	:param str: The input vector layer of aggregated or disaggregated administrative units
	:param int: The identifier column of the administrative units
	'''
	gscript.run_command('v.to.rast', input= vector, type='area', output=vector+'_rast', use='cat', overwrite=True)
	gscript.run_command('r.to.vect', input=vector+'_rast', output= vector+'_rastTovect', type='area', column=id, flags='v',overwrite=True)
	gscript.run_command('v.db.join', map=vector+'_rastTovect', column='cat',other_table=vector, other_column='cat')
	
def prediction_calc(vector, up_vector, column, up_column,weight):
	'''
	Creates a raster of predicted data and recalculates them inside each disaggregated administrative unit
	using the sum function of v.rast.stats
	
	:param str: the input vector layer of disaggregated adlministrative units
	:param str: the input vector layer of aggregated administrative units
	:param str: observed data's column of the level-down vector
	:param str: observed data's column of the level-up vector
	:param str: name of input weighting layer
	'''
	gscript.run_command('v.area.weigh', vector=up_vector, weight=weight, output='Higher_level_pop', column=up_column,overwrite=True)
	gscript.run_command('v.rast.stats', map=vector+'_rastTovect', raster='Higher_level_pop', column_prefix=column, method='sum')
	
def attrTable_To_df(vector):
	'''
	Exports an attribute table to a csv file and converts it to a dataframe
	
	:param str: name of vector layer
	'''
	## Access to GRASS database
	sqlpath = gscript.read_command("db.databases", driver="sqlite").replace('\r\n', '')
	con = sqlite3.connect(sqlpath)
	sqlstat="SELECT * FROM "+vector+";"
	df = pd.read_sql_query(sqlstat, con)
	con.close()
	return df
	

		
	
def Error_calc(df, column,id,vector,output):
	'''
	Calculates the error of prediction and generates a difference map ((observed data - predicted data)/observed data)
	
	:param dataframe: represents the attribute table containing the observed cencus data and the predicted ones
	:param str: name of observed census data column
	:param int: the identifier column of the administrative units 
	:param str: name of vector layer of disaggregated administrative units
	:param str: name of output prediction error map 
	'''
	observed = df.loc[:,column]
	predicted = df.loc[:,column+'_sum']
	error_list = []
	for i in range(len(observed.values)):
		error = ((predicted.values[i] - observed.values[i])/ observed.values[i])*100
		error_list.append(error)
	df1 = df[id]
	df2 = pd.DataFrame(error_list, columns=['error'])
	df_error = pd.concat((df1,df2), axis=1)
	outputfile=tempfile.gettempdir()
	df_error.to_csv(path_or_buf=outputfile+'\\error.csv') 

	# create a csvt file containing columns types
	f = open(outputfile+'\\error.csvt', 'w')
	header_string = '"Integer", "Integer","Real"'
	f.write(header_string)
	f.close()

	#importing the csv to the database
	gscript.run_command('db.in.ogr', input=outputfile+'\\error.csv', output='error_table_in', overwrite=True)
	# join the two tables
	gscript.run_command('v.db.join', map=vector+'_rastTovect', column=id, other_table='error_table_in', other_column=id)	
	gscript.run_command('v.to.rast', input=vector+'_rastTovect', type='area', output=output,use='attr', attribute_column='error', overwrite=True)
	os.remove(outputfile+'\\error.csv')
	os.remove(outputfile+'\\error.csvt')
  
	
def RMSE(df,column):
	'''
	Calculates the root mean square error 
	
	:param dataframe: represents the attribute table containing the observed cencus data and the predicted ones
	:param str: name of observed census data column
	'''
	observed = df.loc[:,column]
	predicted = df.loc[:,column+'_sum']
	s = 0
	somme = 0
	n = len(observed.values)
	for i in range(n):
		somme += observed.values[i]
		s+=((predicted.values[i]) -(observed.values[i])) **2
	rmse = np.sqrt(s/n)
	rmse_pourcent = (rmse / (somme/n))*100
	return rmse,rmse_pourcent
	
def MAE(df,column):
	'''
	Calculates the mean absolute error 
	
	:param dataframe: represents the attribute table containing the observed cencus data and the predicted ones
	:param str: name of observed census data column
	'''
	observed = df.loc[:,column]
	predicted = df.loc[:,column+'_sum']
	s = 0
	somme = 0
	n = len(observed.values)
	for i in range(n):
		somme += observed.values[i]
		s+=np.abs((predicted.values[i]) -(observed.values[i]))
	mae = s/n
	mae_pourcent = (mae / (somme/n))*100
	return mae,mae_pourcent
	
def R_squared(df,column):
	'''
	Calculates the R-squared
	
	:param dataframe: represents the attribute table containing the observed cencus data and the predicted ones
	:param str: name of observed census data column
	'''
	observed = df.loc[:,column]
	predicted = df.loc[:,column+'_sum']
	corr = np.corrcoef(observed,predicted)
	r_squared = corr**2
	return r_squared[0,1]

def df_plot(df,column,plot):
	'''
	Calculates the root mean square error 
	
	:param dataframe: represents the attribute table containing the observed cencus data and the predicted ones
	:param str: name of observed census data column
	'''
	observed = df.loc[:,column]
	predicted = df.loc[:,column+'_sum']

	fig, ax = plt.subplots()
	ax.scatter(observed, predicted)
	ax.plot([observed.min(), observed.max()], [predicted.min(), predicted.max()], 'k--', lw=4)
	ax.set_xlabel('Observed')
	ax.set_ylabel('Predicted')
	plt.savefig(plot+'.png', bbox_inches='tight')
		
def main():
	global df, up_df
	# user's values
	vector = options['vector']
	up_vector = options['up_vector']
	id = options['id']
	column = options['column']
	weight = options['weight'] 
	up_id = options['up_id']
	up_column = options['up_column']
	plot = options['plot']
	output=options['output']
	
	
	# vector exists?
	result = gscript.find_file(vector, element='vector')
	if len(result['name']) == 0:
		gscript.fatal(_("Input vector <%s> not found") % vector)
    
		
	# id column exists ?
	if id not in gscript.vector_columns(vector).keys():
		gscript.fatal(_("<%s> column does not exist for vector %s") % (id, vector))

    # is  id column numeric?
	coltype = gscript.vector_columns(vector)[id]['type']
	
	if coltype not in ('INTEGER'):
		gscript.fatal(_("<%s> column must be Integer")% (id))
	
	# column exists ?
	if column not in gscript.vector_columns(vector).keys():
		gscript.fatal(_("<%s> column does not exist for vector %s") % (column, vector))
	
	# is column numeric?
	coltype = gscript.vector_columns(vector)[column]['type']
	
	if coltype not in ('INTEGER', 'DOUBLE PRECISION'):
		gscript.fatal(_("<%s> column must be numeric")% (column))	
	
	
	# up_vector exists?
	result = gscript.find_file(up_vector, element='vector')
	if len(result['name']) == 0:
		gscript.fatal(_("Input vector <%s> not found") % up_vector)	
	
	# weight exists?

	result = gscript.find_file(weight, element='cell')
	if (len(result['name']) == 0):
		gscript.fatal(_("Input weight <%s> not found") % weight)
		
	# up_id column exists ?
	if up_id not in gscript.vector_columns(up_vector).keys():
		gscript.fatal(_("<%s> column does not exist for vector %s") % (up_id, up_vector))

    # is  up_id column numeric?
	coltype = gscript.vector_columns(up_vector)[up_id]['type']
	
	if coltype not in ('INTEGER'):
		gscript.fatal(_("<%s> column must be Integer")% (up_id))
	
	# up_column exists ?
	if up_column not in gscript.vector_columns(up_vector).keys():
		gscript.fatal(_("<%s> column does not exist for vector %s") % (up_column, up_vector))
	
	# is  up_column numeric?
	coltype = gscript.vector_columns(up_vector)[up_column]['type']
	
	if coltype not in ('INTEGER', 'DOUBLE PRECISION'):
		gscript.fatal(_("<%s> column must be numeric")% (up_column))	
	
	# set the region and the resolution 
	gscript.run_command('g.region', raster= weight, res= gscript.raster_info(weight).nsres)
	# prepare data 
	data_preparation(vector.split("@")[0],id)
	data_preparation(up_vector.split("@")[0],up_id)
	# calculate the predicted data 
	prediction_calc(vector.split("@")[0], up_vector.split("@")[0],column, up_column, weight)
	# export the attribute table to a dataframe
	df = attrTable_To_df(vector.split("@")[0]+'_rastTovect')
	# calculate the %error
	Error_calc(df, column,id,vector.split("@")[0],output)
	# calculate the RMSE
	rmse = RMSE(df,column)[1]
	# calculate the MAE
	mae = MAE(df,column)[1]
	# calculate the R²
	r = R_squared(df,column)
	# plot
	df_plot(df,column,plot)
	print('RMSE',rmse)
	print('MAE',mae)
	print('R_square',r)
	
		










 

if __name__ == "__main__":
	options, flags = gscript.parser()
	main()

