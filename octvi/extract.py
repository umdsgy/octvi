## set up logging
import logging, os
logging.basicConfig(level=os.environ.get("LOGLEVEL","INFO"))
log = logging.getLogger(__name__)

## import modules
import octvi.exceptions, octvi.array
import numpy as np
import h5py

# Import pyhdf for HDF4 files
try:
    from pyhdf.SD import SD, SDC
    PYHDF_AVAILABLE = True
except ImportError:
    log.warning("pyhdf not available. HDF4 file support will be limited.")
    PYHDF_AVAILABLE = False


def getDatasetNames(stack_path:str) -> list:
	"""
	Returns list of all subdataset names, in format
	suitable for passing to other functions'
	'dataset_name' argument
	"""

	ext = os.path.splitext(stack_path)[1]
	
	if ext == ".hdf":
		# Use pyhdf for HDF4
		if not PYHDF_AVAILABLE:
			raise ImportError("pyhdf is required to read HDF4 files. Install with: pip install pyhdf")
		
		hdf = SD(stack_path, SDC.READ)
		datasets = hdf.datasets()
		dataset_names = list(datasets.keys())
		hdf.end()
		return dataset_names
		
	elif ext == ".h5":
		# Use h5py for HDF5
		dataset_names = []
		with h5py.File(stack_path, 'r') as f:
			# VIIRS structure: /HDFEOS/GRIDS/[GridName]/Data Fields/[dataset_name]
			if 'HDFEOS' in f and 'GRIDS' in f['HDFEOS']:
				grid_names = list(f['HDFEOS']['GRIDS'].keys())
				if grid_names:
					grid_name = grid_names[0]
					if 'Data Fields' in f[f'HDFEOS/GRIDS/{grid_name}']:
						data_fields = f[f'HDFEOS/GRIDS/{grid_name}/Data Fields']
						dataset_names = list(data_fields.keys())
			else:
				# Fallback: search for all datasets
				def get_datasets(name, obj):
					if isinstance(obj, h5py.Dataset):
						dataset_names.append(name.split('/')[-1])
				f.visititems(get_datasets)
		return dataset_names
	else:
		raise octvi.exceptions.FileTypeError("File must be of format .hdf or .h5")


def datasetToArray(stack_path, dataset_name) -> "numpy array":
	"""
	This function extracts a specified subdataset from a hierarchical format
	(HDF4 or HDF5) and returns it as a numpy array.

	...

	Parameters
	----------

	stack_path: str
		Full path to hierarchical file containing the desired subdataset
	dataset_name: str
		Name of desired subdataset, as it appears in the hierarchical file
	"""

	ext = os.path.splitext(stack_path)[1]
	
	if ext == ".hdf":
		# Use pyhdf for HDF4 files
		if not PYHDF_AVAILABLE:
			raise ImportError("pyhdf is required to read HDF4 files. Install with: pip install pyhdf")
		
		try:
			hdf = SD(stack_path, SDC.READ)
			
			# List available datasets for debugging
			available_datasets = list(hdf.datasets().keys())
			
			if dataset_name not in available_datasets:
				hdf.end()
				raise octvi.exceptions.DatasetNotFoundError(
					f"Dataset '{dataset_name}' not found in '{os.path.basename(stack_path)}'. "
					f"Available datasets: {available_datasets}"
				)
			
			dataset = hdf.select(dataset_name)
			data = dataset.get()
			dataset.endaccess()
			hdf.end()
			
			return data
			
		except Exception as e:
			log.error(f"Error reading dataset '{dataset_name}' from HDF4 file: {e}")
			raise
		
	elif ext == ".h5":
		# Use h5py for HDF5 files
		try:
			with h5py.File(stack_path, 'r') as f:
				# Try VIIRS structure first: /HDFEOS/GRIDS/[GridName]/Data Fields/[dataset_name]
				if 'HDFEOS' in f and 'GRIDS' in f['HDFEOS']:
					grid_names = list(f['HDFEOS']['GRIDS'].keys())
					if grid_names:
						grid_name = grid_names[0]
						data_fields_path = f'HDFEOS/GRIDS/{grid_name}/Data Fields'
						if data_fields_path in f and dataset_name in f[data_fields_path]:
							data = f[f'{data_fields_path}/{dataset_name}'][:]
							return data
				
				# Fallback: search for dataset by name
				dataset_path = None
				def find_dataset(name, obj):
					nonlocal dataset_path
					if isinstance(obj, h5py.Dataset) and name.split('/')[-1] == dataset_name:
						dataset_path = name
				
				f.visititems(find_dataset)
				
				if dataset_path:
					data = f[dataset_path][:]
					return data
				else:
					raise octvi.exceptions.DatasetNotFoundError(
						f"Dataset '{dataset_name}' not found in '{os.path.basename(stack_path)}'"
					)
					
		except octvi.exceptions.DatasetNotFoundError:
			raise
		except Exception as e:
			log.error(f"Error reading dataset '{dataset_name}' from HDF5 file: {e}")
			raise
	else:
		raise octvi.exceptions.FileTypeError("File must be of format .hdf or .h5")


def datasetToPath(stack_path, dataset_name) -> str:
	"""
	Returns the full path to a subdataset within a hierarchical file.
	For HDF4, returns GDAL subdataset path. For HDF5, returns h5py-style path.
	"""
	ext = os.path.splitext(stack_path)[1]
	
	if ext == ".hdf":
		# For HDF4, we still use GDAL subdataset notation for compatibility
		# even though we read data with pyhdf
		from osgeo import gdal
		ds = gdal.Open(stack_path, 0)
		if ds is None:
			raise octvi.exceptions.FileTypeError(f"Failed to open HDF4 file: {stack_path}")
		
		for sd in ds.GetSubDatasets():
			sdName = sd[0].split(":")[-1].strip('"')
			if sdName == dataset_name:
				ds = None
				return sd[0]
		
		ds = None
		raise octvi.exceptions.DatasetNotFoundError(
			f"Dataset '{dataset_name}' not found in '{os.path.basename(stack_path)}'"
		)
		
	elif ext == ".h5":
		# For HDF5, return h5py path
		with h5py.File(stack_path, 'r') as f:
			# Try VIIRS structure
			if 'HDFEOS' in f and 'GRIDS' in f['HDFEOS']:
				grid_names = list(f['HDFEOS']['GRIDS'].keys())
				if grid_names:
					grid_name = grid_names[0]
					data_fields_path = f'HDFEOS/GRIDS/{grid_name}/Data Fields'
					if data_fields_path in f and dataset_name in f[data_fields_path]:
						return f'{data_fields_path}/{dataset_name}'
			
			# Fallback: search for dataset
			dataset_path = None
			def find_dataset(name, obj):
				nonlocal dataset_path
				if isinstance(obj, h5py.Dataset) and name.split('/')[-1] == dataset_name:
					dataset_path = name
			
			f.visititems(find_dataset)
			
			if dataset_path:
				return dataset_path
			else:
				raise octvi.exceptions.DatasetNotFoundError(
					f"Dataset '{dataset_name}' not found in '{os.path.basename(stack_path)}'"
				)
	else:
		raise octvi.exceptions.FileTypeError("File must be of format .hdf or .h5")


def datasetToRaster(stack_path,dataset_name, out_path,dtype = None, *args, **kwargs) -> None:
	"""
	Wrapper for datasetToArray and arrayToRaster which pulls
	subdataset from hdf or h5 file and saves to new location.

	...

	Arguments
	---------

	stack_path: str
	dataset_name: str
	out_path: str

	"""

	sd_array = datasetToArray(stack_path, dataset_name)
	return octvi.array.toRaster(sd_array, out_path, model_file = stack_path,dtype=dtype)


def ndviToArray(in_stack) -> "numpy array":
	"""
	This function finds the correct Red and NIR bands
	from a hierarchical file, calculates an NDVI array,
	and returns the output in numpy array format.

	Valid input formats are MODIS HDF or VIIRS HDF5 (h5).

	...

	Parameters
	----------

	in_stack: str
		Full path to input hierarchical file

	"""

	suffix = os.path.basename(in_stack).split(".")[0][3:7]

	# check whether it's an ndvi product
	if suffix == "09Q4" or suffix == "13Q4":
		arr_ndvi = datasetToArray(in_stack, "250m 8 days NDVI")

	elif suffix == "13Q1":
		arr_ndvi = datasetToArray(in_stack, "250m 16 days NDVI")

	elif suffix == "09CM":
		## determine correct band subdataset names
		ext = os.path.splitext(in_stack)[1]
		if ext == ".hdf":
			sdName_red = "Coarse Resolution Surface Reflectance Band 1"
			sdName_nir = "Coarse Resolution Surface Reflectance Band 2"
		elif ext == '.h5':
			sdName_red = "SurfReflect_I1"
			sdName_nir = "SurfReflect_I2"

		## extract red and nir bands from stack
		arr_red = datasetToArray(in_stack,sdName_red)
		arr_nir = datasetToArray(in_stack,sdName_nir)

		## perform calculation
		arr_ndvi = octvi.array.calcNdvi(arr_red,arr_nir)

	else:
		## determine correct band subdataset names
		ext = os.path.splitext(in_stack)[1]
		if ext == ".hdf":
			sdName_red = "sur_refl_b01"
			sdName_nir = "sur_refl_b02"
		elif ext == ".h5":
			if suffix == "09A1":  # VNP09A1 / VJ109A1 use M-bands at 1km
				sdName_red = "SurfReflect_M5"
				sdName_nir = "SurfReflect_M7"
			else:  # VNP09H1 uses I-bands at 500m
				sdName_red = "SurfReflect_I1"
				sdName_nir = "SurfReflect_I2"
		else:
			raise octvi.exceptions.FileTypeError("File must be of type .hdf or .h5")

		## extract red and nir bands from stack
		arr_red = datasetToArray(in_stack,sdName_red)
		arr_nir = datasetToArray(in_stack,sdName_nir)

		## perform calculation
		arr_ndvi = octvi.array.calcNdvi(arr_red,arr_nir)

	return arr_ndvi


def gcviToArray(in_stack:str) -> "numpy array":
	"""
	This function finds the correct Green and NIR bands
	from a hierarchical file, calculates a GCVI array,
	and returns the output in numpy array format.

	Valid input format is MOD09CMG HDF.

	...

	Parameters
	----------

	in_stack: str
		Full path to input hierarchical file

	"""

	suffix = os.path.basename(in_stack).split(".")[0][3:7]

	# check whether it's an ndvi product
	if suffix  == "09CM":
		## determine correct band subdataset names
		ext = os.path.splitext(in_stack)[1]
		if ext == ".hdf":
			sdName_green = "Coarse Resolution Surface Reflectance Band 4"
			sdName_nir = "Coarse Resolution Surface Reflectance Band 2"
		elif ext == '.h5':
			sdName_green = "SurfReflect_M4"
			sdName_nir = "SurfReflect_I2"

		## extract red and nir bands from stack
		arr_green = datasetToArray(in_stack,sdName_green)
		arr_nir = datasetToArray(in_stack,sdName_nir)

		## perform calculation
		arr_gcvi = octvi.array.calcGcvi(arr_green,arr_nir)

	elif suffix == "09A1":
		ext = os.path.splitext(in_stack)[1]
		if ext == ".hdf":  # MOD09A1
			sdName_green = "sur_refl_b04"
			sdName_nir = "sur_refl_b02"
		elif ext == ".h5":  # VNP09A1 / VJ109A1 use M-bands at 1km
			sdName_green = "SurfReflect_M4"
			sdName_nir = "SurfReflect_M7"
		else:
			raise octvi.exceptions.FileTypeError("File must be of format .hdf or .h5")
		arr_green = datasetToArray(in_stack, sdName_green)
		arr_nir = datasetToArray(in_stack, sdName_nir)
		arr_gcvi = octvi.array.calcGcvi(arr_green, arr_nir)

	else:
		raise octvi.exceptions.UnsupportedError("Only MOD09CMG, MOD09A1, VNP09A1, and VJ109A1 imagery is supported for GCVI generation")

	return arr_gcvi


def ndwiToArray(in_stack:str) -> "numpy array":
	"""
	This function finds the correct SWIR and NIR bands
	from a hierarchical file, calculates a NDWI array,
	and returns the output in numpy array format.

	Valid input format is HDF.

	...

	Parameters
	----------

	in_stack: str
		Full path to input hierarchical file

	"""

	suffix = os.path.basename(in_stack).split(".")[0][3:7]

	if suffix == "09A1":
		sdName_nir = "sur_refl_b02"
		sdName_swir = "sur_refl_b05"
		arr_nir = datasetToArray(in_stack, sdName_nir)
		arr_swir = datasetToArray(in_stack,sdName_swir)
		arr_ndwi = octvi.array.calcNdwi(arr_nir,arr_swir)
	else:
		raise octvi.exceptions.UnsupportedError("Only MOD09A1 imagery is supported for NDWI generation")

	return arr_ndwi


def ndviToRaster(in_stack,out_path,qa_name=None) -> str:
	"""
	This function directly converts a hierarchical data
	file into an NDVI raster.

	Returns the string path to the output file

	***

	Parameters
	----------
	in_stack:str
	out_path:str
	qa_name (optional):str
		Name of QA dataset, if included produces
		two-band tiff
	"""

	# create ndvi array
	ndviArray = ndviToArray(in_stack)

	# apply cloud, shadow, and water masks
	ndviArray = octvi.array.mask(ndviArray, in_stack)

	if qa_name is None:
		octvi.array.toRaster(ndviArray,out_path,in_stack)
	else:
		# get qa array
		qaArray = datasetToArray(in_stack,qa_name)
		# create multiband at out_path
		octvi.array.toRaster(ndviArray,out_path,in_stack,qa_array = qaArray)

	return out_path


def gcviToRaster(in_stack:str,out_path:str) -> str:
	"""
	This function directly converts a hierarchical data
	file into a GCVI raster.

	Returns the string path to the output file
	"""

	# create gcvi array
	gcviArray = gcviToArray(in_stack)

	# apply cloud, shadow, and water masks
	gcviArray = octvi.array.mask(gcviArray, in_stack)

	octvi.array.toRaster(gcviArray,out_path,in_stack)

	return out_path


def ndwiToRaster(in_stack:str, out_path:str) -> str:
	"""
	This function directly converts a hierarchical data
	file into an NDWI raster.

	Returns the string path to the output file
	"""

	# create ndwi array
	ndwiArray = ndwiToArray(in_stack)

	# apply cloud, shadow, and water masks
	ndwiArray = octvi.array.mask(ndwiArray, in_stack)

	octvi.array.toRaster(ndwiArray,out_path,in_stack)

	return out_path


def cmgToViewAngArray(source_stack,product="MOD09CMG") -> "numpy array":
	"""
	This function takes the path to a M*D CMG file, and returns
	the view angle of each pixel. Ephemeral water pixels are
	set to 999, to be used as a last resort in compositing.

	Returns a numpy array of the same dimensions as the input raster.

	***

	Parameters
	----------
	source_stack:str
		Path to the M*D CMG .hdf file on disk
	"""
	if product == "MOD09CMG":
		vang_arr = datasetToArray(source_stack,"Coarse Resolution View Zenith Angle")
		state_arr = datasetToArray(source_stack,"Coarse Resolution State QA")
		water = ((state_arr & 0b111000)) # check bits
		vang_arr[water==32]=9999 # ephemeral water???
		vang_arr[vang_arr<=0]=9999
	elif product == "VNP09CMG":
		vang_arr = datasetToArray(source_stack,"SensorZenith")
		vang_arr[vang_arr<=0]=9999
	return vang_arr


def cmgListToWaterArray(stacks:list,product="MOD09CMG") -> "numpy array":
	"""
	This function takes a list of CMG .hdf files, and returns
	a binary array, with "0" for non-water pixels and "1" for
	water pixels. If any file flags water in a pixel, its value
	is stored as "1"

	***

	Parameters
	----------
	stacks:list
		List of hdf filepaths (M*D**CMG)
	"""
	water_list = []
	for source_stack in stacks:
		if product == "MOD09CMG":
			state_arr = datasetToArray(source_stack,"Coarse Resolution State QA")
			water = ((state_arr & 0b111000)) # check bits
			water[water==56]=1 # deep ocean
			water[water==48]=1 # continental/moderate ocean
			water[water==24]=1 # shallow inland water
			water[water==40]=1 # deep inland water
			water[water==0]=1 # shallow ocean
			water[state_arr==0]=0
			water[water!=1]=0 # set non-water to zero
		elif product == "VNP09CMG":
			state_arr = datasetToArray(source_stack,"State_QA")
			water = ((state_arr & 0b111000)) # check bits 3-5
			water[water == 40] = 0 # "coastal" = 101
			water[water>8]=1 # sea water = 011; inland water = 010
			water[water!=1]=0 # set non-water to zero
			water[water!=0]=1
		water_list.append(water)
	water_final = np.maximum.reduce(water_list)
	return water_final


def cmgToRankArray(source_stack,product="MOD09CMG") -> "numpy array":
	"""
	This function takes the path to a MOD**CMG file, and returns
	the rank of each pixel, as defined on page 7 of the MOD09 user
	guide (http://modis-sr.ltdri.org/guide/MOD09_UserGuide_v1.4.pdf)

	Returns a numpy array of the same dimensions as the input raster

	***

	Parameters
	----------
	source_stack:str
		Path to the CMG .hdf/.h5 file on disk
	product:str
		String of either MOD09CMG or VNP09CMG
	"""
	if product == "MOD09CMG":
		qa_arr = datasetToArray(source_stack,"Coarse Resolution QA")
		state_arr = datasetToArray(source_stack,"Coarse Resolution State QA")
		vang_arr = datasetToArray(source_stack,"Coarse Resolution View Zenith Angle")
		vang_arr[vang_arr<=0]=9999
		sang_arr = datasetToArray(source_stack,"Coarse Resolution Solar Zenith Angle")
		rank_arr = np.full(qa_arr.shape,10) # empty rank array

		## perform the ranking!
		logging.debug("--rank 9: SNOW")
		SNOW = ((state_arr & 0b1000000000000) | (state_arr & 0b1000000000000000)) # state bit 12 OR 15
		rank_arr[SNOW>0]=9 # snow
		del SNOW
		logging.debug("--rank 8: HIGHAEROSOL")
		HIGHAEROSOL=(state_arr & 0b11000000) # state bits 6 AND 7
		rank_arr[HIGHAEROSOL==192]=8
		del HIGHAEROSOL
		logging.debug("--rank 7: CLIMAEROSOL")
		CLIMAEROSOL=(state_arr & 0b11000000) # state bits 6 & 7
		#CLIMAEROSOL=(cloudMask & 0b100000000000000) # cloudMask bit 14
		rank_arr[CLIMAEROSOL==0]=7 # default aerosol level
		del CLIMAEROSOL
		logging.debug("--rank 6: UNCORRECTED")
		UNCORRECTED = (qa_arr & 0b11) # qa bits 0 AND 1
		rank_arr[UNCORRECTED==3]=6 # flagged uncorrected
		del UNCORRECTED
		logging.debug("--rank 5: SHADOW")
		SHADOW = (state_arr & 0b100) # state bit 2
		rank_arr[SHADOW==4]=5 # cloud shadow
		del SHADOW
		logging.debug("--rank 4: CLOUDY")
		# set adj to 11 and internal to 12 to verify in qa output
		CLOUDY = ((state_arr & 0b11)) # state bit 0 OR bit 1 OR bit 10 OR bit 13
		#rank_arr[CLOUDY!=0]=4 # cloud pixel
		del CLOUDY
		CLOUDADJ = (state_arr & 0b10000000000000)
		#rank_arr[CLOUDADJ>0]=4 # adjacent to cloud
		del CLOUDADJ
		CLOUDINT = (state_arr & 0b10000000000)
		rank_arr[CLOUDINT>0]=4
		del CLOUDINT
		logging.debug("--rank 3: HIGHVIEW")
		rank_arr[sang_arr>(85/0.01)]=3 # HIGHVIEW
		logging.debug("--rank 2: LOWSUN")
		rank_arr[vang_arr>(60/0.01)]=2 # LOWSUN
		# BAD pixels
		logging.debug("--rank 1: BAD pixels") # qa bits (2-5 OR 6-9 == 1110)
		BAD = ((qa_arr & 0b111100) | (qa_arr & 0b1110000000))
		rank_arr[BAD==112]=1
		rank_arr[BAD==896]=1
		rank_arr[BAD==952]=1
		del BAD

		logging.debug("-building water mask")
		water = ((state_arr & 0b111000)) # check bits
		water[water==56]=1 # deep ocean
		water[water==48]=1 # continental/moderate ocean
		water[water==24]=1 # shallow inland water
		water[water==40]=1 # deep inland water
		water[water==0]=1 # shallow ocean
		rank_arr[water==1]=0
		vang_arr[water==32]=9999 # ephemeral water???
		water[state_arr==0]=0
		water[water!=1]=0 # set non-water to zero

	elif product == "VNP09CMG":
		qf2 = datasetToArray(source_stack,"SurfReflect_QF2")
		qf4 = datasetToArray(source_stack,"SurfReflect_QF4")
		state_arr = datasetToArray(source_stack,"State_QA")
		vang_arr = datasetToArray(source_stack,"SensorZenith")
		vang_arr[vang_arr<=0]=9999
		sang_arr = datasetToArray(source_stack,"SolarZenith")
		rank_arr = np.full(state_arr.shape,10) # empty rank array

		## perform the ranking!
		logging.debug("--rank 9: SNOW")
		SNOW = (state_arr & 0b1000000000000000) # state bit 15
		rank_arr[SNOW>0]=9 # snow
		del SNOW
		logging.debug("--rank 8: HIGHAEROSOL")
		HIGHAEROSOL=(qf2 & 0b10000) # qf2 bit 4
		rank_arr[HIGHAEROSOL!=0]=8
		del HIGHAEROSOL
		logging.debug("--rank 7: AEROSOL")
		CLIMAEROSOL=(state_arr & 0b1000000) # state bit 6
		#CLIMAEROSOL=(cloudMask & 0b100000000000000) # cloudMask bit 14
		#rank_arr[CLIMAEROSOL==0]=7 # "No"
		del CLIMAEROSOL
		# logging.debug("--rank 6: UNCORRECTED")
		# UNCORRECTED = (qa_arr & 0b11) # qa bits 0 AND 1
		# rank_arr[UNCORRECTED==3]=6 # flagged uncorrected
		# del UNCORRECTED
		logging.debug("--rank 5: SHADOW")
		SHADOW = (state_arr & 0b100) # state bit 2
		rank_arr[SHADOW!=0]=5 # cloud shadow
		del SHADOW
		logging.debug("--rank 4: CLOUDY")
		# set adj to 11 and internal to 12 to verify in qa output
		# CLOUDY = ((state_arr & 0b11)) # state bit 0 OR bit 1 OR bit 10 OR bit 13
		# rank_arr[CLOUDY!=0]=4 # cloud pixel
		# del CLOUDY
		# CLOUDADJ = (state_arr & 0b10000000000) # nonexistent for viirs
		# #rank_arr[CLOUDADJ>0]=4 # adjacent to cloud
		# del CLOUDADJ
		CLOUDINT = (state_arr & 0b10000000000) # state bit 10
		rank_arr[CLOUDINT>0]=4
		del CLOUDINT
		logging.debug("--rank 3: HIGHVIEW")
		rank_arr[sang_arr>(85/0.01)]=3 # HIGHVIEW
		logging.debug("--rank 2: LOWSUN")
		rank_arr[vang_arr>(60/0.01)]=2 # LOWSUN
		# BAD pixels
		logging.debug("--rank 1: BAD pixels") # qa bits (2-5 OR 6-9 == 1110)
		BAD = (qf4 & 0b110)
		rank_arr[BAD!= 0]=1
		del BAD

		logging.debug("-building water mask")
		water = ((state_arr & 0b111000)) # check bits 3-5
		water[water == 40] = 0 # "coastal" = 101
		water[water>8]=1 # sea water = 011; inland water = 010
		# water[water==16]=1 # inland water = 010
		# water[state_arr==0]=0
		water[water!=1]=0 # set non-water to zero
		water[water!=0]=1
		rank_arr[water==1]=0

	# return the results
	return rank_arr


def cmgBestViPixels(input_stacks:list,vi="NDVI",product = "MOD09CMG",snow_mask=False) -> "numpy array":
	"""
	This function takes a list of hdf stack paths, and
	returns the 'best' VI value for each pixel location,
	determined through the ranking method (see
	cmgToRankArray() for details).

	***

	Parameters
	----------
	input_stacks:list
		A list of strings, each pointing to a CMG hdf/h5 file
		on disk
	product:str
		A string of either "MOD09CMG" or "VNP09CMG"
	"""

	viExtractors = {
		"NDVI":ndviToArray,
		"GCVI":gcviToArray
	}

	rankArrays = [cmgToRankArray(hdf,product) for hdf in input_stacks]
	vangArrays = [cmgToViewAngArray(hdf,product) for hdf in input_stacks]
	try:
		viArrays = [viExtractors[vi](hdf) for hdf in input_stacks]
	except KeyError:
		raise octvi.exceptions.UnsupportedError(f"Index type '{vi}' is not recognized or not currently supported.")
	# no nodata wanted
	for i in range(len(rankArrays)):
		rankArrays[i][viArrays[i] == -3000] = 0

	# apply snow mask if requested
	if snow_mask:
		for rankArray in rankArrays:
			rankArray[rankArray==9] = 0

	idealRank = np.maximum.reduce(rankArrays)

	# mask non-ideal view angles
	for i in range(len(vangArrays)):
		vangArrays[i][rankArrays[i] != idealRank] = 9998
		vangArrays[i][vangArrays[i] == 0] = 9997

	idealVang = np.minimum.reduce(vangArrays)

	finalVi = np.full(viArrays[0].shape,-3000)

	# mask each viArray to only where it matches ideal rank
	for i in range(len(viArrays)):
		finalVi[vangArrays[i] == idealVang] = viArrays[i][vangArrays[i] == idealVang]

	# mask out ranks that are too low
	finalVi[idealRank <=7] = -3000

	# mask water
	water = cmgListToWaterArray(input_stacks,product)
	finalVi[water==1] = -3000

	# return result
	return finalVi


def qaTo8BitArray(stack_path) -> "numpy array":
	"""Returns an 8-bit QA array for the passed image file

	MODIS and VIIRS use 16-bit QA layers, but many of those bits
	are redundant or unnecessary for purposes of VI mapping. For
	example, non-land pixels are masked by default, so the land/
	water flag is unused.

	This function pares down the 16-bit mask into an 8-bit
	version that retains all necessary functionality.

	***

	Parameters
	----------
	stack_path:str
		Full path to input hierarchical file on disk

	"""
	log.warning("octvi.extract.qaTo8BitArray() is not implemented!")
	return None
