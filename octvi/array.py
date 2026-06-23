## set up logging
import logging, os
logging.basicConfig(level=os.environ.get("LOGLEVEL","INFO"))
log = logging.getLogger(__name__)

import h5py, octvi.extract
import numpy as np
from osgeo import gdal
from osgeo.gdal_array import *

# Import pyhdf for HDF4 files
try:
    from pyhdf.SD import SD, SDC
    PYHDF_AVAILABLE = True
except ImportError:
    log.warning("pyhdf not available. HDF4 file support will be limited.")
    PYHDF_AVAILABLE = False

supported_indices = ["NDVI","GCVI","NDWI"]

def calcNdvi(red_array,nir_array) -> "numpy array":
	"""
	A function to robustly build an NDVI array from two
	arrays (red and NIR) of the same shape.

	Resulting array is scaled by 10000, with values stored
	as integers. Nodata value is -3000.

	...

	Parameters
	----------

	red_array: numpy.array
		Array of red reflectances
	nir_array: numpy.array
		Array of near-infrared reflectances

	"""

	## perform NDVI generation
	ndvi = np.divide((nir_array - red_array),(nir_array + red_array))

	## rescale and replace infinities
	ndvi = ndvi * 10000
	ndvi[ndvi == np.inf] = -3000
	ndvi[ndvi == -np.inf] = -3000
	ndvi = ndvi.astype(int)

	## return array
	return ndvi

def calcGcvi(green_array,nir_array) -> "numpy array":
	"""
	A function to robustly build a GCVI array from two
	arrays (green and NIR) of the same shape.

	Resulting array is scaled by 10000, with values stored
	as integers. Nodata value is -3000.

	...

	Parameters
	----------

	green_array: numpy.array
		Array of green reflectances
	nir_array: numpy.array
		Array of near-infrared reflectances

	"""

	## perform NDVI generation
	gcvi = np.divide(nir_array, green_array) - 1

	## rescale and replace infinities
	gcvi = gcvi * 10000
	gcvi[gcvi == np.inf] = -3000
	gcvi[gcvi == -np.inf] = -3000
	gcvi = gcvi.astype(int)

	## return array
	return gcvi

def calcNdwi(nir_array, swir_array) -> "numpy array":
	"""
	A function to robustly build an NDWI array from two
	arrays (SWIR and NIR) of the same shape.

	Resulting array is scaled by 10000, with values stored
	as integers. Nodata value is -3000.

	...

	Parameters
	----------

	nir_array: numpy.array
		Array of near-infrared reflectances
	swir_array: numpy.array
		Array of shortwave infrared reflectances

	"""

	## perform NDVI generation
	ndwi = np.divide((nir_array - swir_array),(nir_array + swir_array))

	## rescale and replace infinities
	ndwi = ndwi * 10000
	ndwi[ndwi == np.inf] = -3000
	ndwi[ndwi == -np.inf] = -3000
	ndwi = ndwi.astype(int)

	## return array
	return ndwi

def mask(in_array, source_stack) -> "numpy array":
	"""
	This function removes non-clear pixels from an input array,
	including clouds, cloud shadow, and water.

	For M*D CMG files, removes pixels ranked below "8" in
	MOD13Q1 compositing method, as well as water.

	Returns a cleaned array.

	...

	Parameters
	----------

	in_array: numpy.array
		The array to be cleaned. This must have the same dimensions
		as source_stack, and preferably have been extracted from the
		stack.
	source_stack: str
		Path to a hierarchical data file containing QA layers with
		which to perform the masking. Currently valid formats include
		MOD09Q1 hdf and VNP09H1 files.
	"""

	## get file extension and product suffix
	ext = os.path.splitext(source_stack)[1]
	suffix = os.path.basename(source_stack).split(".")[0][3:7]


	## product-conditional behavior

	# MODIS pre-generated VI masking
	if suffix == "13Q1" or suffix == "13Q4":
		if suffix[-1] == "1":
			pr_arr = octvi.extract.datasetToArray(source_stack, "250m 16 days pixel reliability")
			qa_arr = octvi.extract.datasetToArray(source_stack, "250m 16 days VI Quality")
		else:
			pr_arr = octvi.extract.datasetToArray(source_stack, "250m 8 days pixel reliability")
			qa_arr = octvi.extract.datasetToArray(source_stack, "250m 8 days VI Quality")


		#in_array[(pr_arr != 0) & (pr_arr != 1)] = -3000

		# mask clouds
		in_array[(qa_arr & 0b11) > 1] = -3000 # bits 0-1 > 01 = Cloudy

		# mask Aerosol
		in_array[(qa_arr & 0b11000000) == 0] = -3000 # climatology
		in_array[(qa_arr & 0b11000000) == 192] = -3000 # high

		# mask water
		in_array[((qa_arr & 0b11100000000000) != 2048) & ((qa_arr & 0b11100000000000) != 4096) & ((qa_arr & 0b11100000000000) != 8192)] = -3000
		# 001 = land, 010 = coastline, 100 = ephemeral water

		# mask snow/ice
		in_array[(qa_arr & 0b100000000000000) != 0] = -3000 # bit 14

		# mask cloud shadow
		in_array[(qa_arr & 0b1000000000000000) != 0] = -3000 # bit 15

		# mask cloud adjacent pixels
		in_array[(qa_arr & 0b100000000) != 0] = -3000 # bit 8

	# MODIS and VIIRS surface reflectance masking
	# CMG
	elif suffix == "09CM":
		if ext == ".hdf": # MOD09CMG
			qa_arr = octvi.extract.datasetToArray(source_stack,"Coarse Resolution QA")
			state_arr = octvi.extract.datasetToArray(source_stack,"Coarse Resolution State QA")
			vang_arr = octvi.extract.datasetToArray(source_stack,"Coarse Resolution View Zenith Angle")
			vang_arr[vang_arr<=0]=9999
			sang_arr = octvi.extract.datasetToArray(source_stack,"Coarse Resolution Solar Zenith Angle")
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
			in_array[rank_arr <= 7] = -3000
		elif ext == ".h5": # VNP09CMG
			qf2 = octvi.extract.datasetToArray(source_stack,"SurfReflect_QF2")
			qf4 = octvi.extract.datasetToArray(source_stack,"SurfReflect_QF4")
			state_arr = octvi.extract.datasetToArray(source_stack,"State_QA")
			vang_arr = octvi.extract.datasetToArray(source_stack,"SensorZenith")
			vang_arr[vang_arr<=0]=9999
			sang_arr = octvi.extract.datasetToArray(source_stack,"SolarZenith")
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
			in_array[rank_arr <= 7] = -3000
		else:
			raise octvi.exceptions.FileTypeError("File must be of format .hdf or .h5")
	# standard
	else:
		# modis
		## MOD09A1
		if suffix == "09A1" and ext == ".hdf":
			qa_arr = octvi.extract.datasetToArray(source_stack, "sur_refl_qc_500m")
			state_arr = octvi.extract.datasetToArray(source_stack,"sur_refl_state_500m")
		## VNP09A1 / VJ109A1 (VIIRS 1km tiled)
		elif suffix == "09A1" and ext == ".h5":
			state_arr = octvi.extract.datasetToArray(source_stack, "SurfReflect_State_1km")
		## all other MODIS products
		elif ext == ".hdf":
			qa_arr = octvi.extract.datasetToArray(source_stack, "sur_refl_qc_250m")
			state_arr = octvi.extract.datasetToArray(source_stack,"sur_refl_state_250m")

		# viirs VNP09H1 (500m tiled)
		elif ext == ".h5":
			qa_arr = octvi.extract.datasetToArray(source_stack, "SurfReflect_QC_500m")
			state_arr = octvi.extract.datasetToArray(source_stack,"SurfReflect_State_500m")

		else:
			raise octvi.exceptions.FileTypeError("File must be of format .hdf or .h5")

		## mask clouds
		in_array[(state_arr & 0b11) != 0 ] = -3000
		in_array[(state_arr & 0b10000000000) != 0] = -3000 # internal cloud mask

		## mask cloud shadow
		in_array[(state_arr & 0b100) != 0] = -3000

		## mask cloud adjacent pixels
		in_array[(state_arr & 0b10000000000000) != 0] = -3000

		## mask aerosols
		in_array[(state_arr & 0b11000000) == 0] = -3000 # climatology
		in_array[(state_arr & 0b11000000) == 192] = -3000 # high; known to be an unreliable flag in MODIS collection 6

		## mask snow/ice
		in_array[(state_arr & 0b1000000000000) != 0] = -3000

		## mask water
		in_array[((state_arr & 0b111000) != 8) & ((state_arr & 0b111000) != 16) & ((state_arr & 0b111000) !=32)] = -3000 # checks against three 'allowed' land/water classes and excludes pixels that don't match

		## mask bad solar zenith
		#in_array[(qa_arr & 0b11100000) != 0] = -3000


	## return output
	return in_array


def _get_hdf4_geotransform(hdf_file):
	"""
	Extract geotransform and projection from HDF4 file using pyhdf.
	
	Parameters
	----------
	hdf_file : str
		Path to HDF4 file
		
	Returns
	-------
	tuple : (geoTransform, projection_wkt, rasterXSize, rasterYSize)
	"""
	if not PYHDF_AVAILABLE:
		raise ImportError("pyhdf is required to read HDF4 files. Install with: pip install pyhdf")
	
	# Open HDF4 file
	hdf = SD(hdf_file, SDC.READ)
	
	# Get global attributes
	attrs = hdf.attributes()
	
	# Default sinusoidal projection for MODIS
	sr = 'PROJCS["unnamed",GEOGCS["Unknown datum based upon the custom spheroid",DATUM["Not specified (based on custom spheroid)",SPHEROID["Custom spheroid",6371007.181,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Sinusoidal"],PARAMETER["longitude_of_center",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["Meter",1]]'
	
	# Try to get StructMetadata
	if 'StructMetadata.0' in attrs:
		struct_meta = attrs['StructMetadata.0']
		
		# Parse StructMetadata for geotransform info
		lines = struct_meta.split('\n')
		ulc_lon = None
		ulc_lat = None
		pixel_size_x = None
		pixel_size_y = None
		x_dim = None
		y_dim = None
		
		for line in lines:
			if 'UpperLeftPointMtrs' in line:
				# Extract coordinates - format: UpperLeftPointMtrs=(lon,lat)
				coords = line.split('=')[1].strip('()\n ')
				try:
					ulc_lon, ulc_lat = map(float, coords.split(','))
				except:
					pass
			elif 'XDim=' in line:
				try:
					x_dim = int(line.split('=')[1].strip())
				except:
					pass
			elif 'YDim=' in line:
				try:
					y_dim = int(line.split('=')[1].strip())
				except:
					pass
		
		# For CMG products, pixel size is 0.05 degrees
		if 'CMG' in os.path.basename(hdf_file):
			pixel_size_x = 0.05
			pixel_size_y = 0.05
			if ulc_lon is not None and ulc_lat is not None:
				# CMG coordinates are in millionths of degrees
				ulc_lon = ulc_lon / 1000000.0
				ulc_lat = ulc_lat / 1000000.0
			# Use WGS84 projection for CMG
			sr = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
		else:
			# For sinusoidal products, calculate pixel size
			# Standard MODIS sinusoidal grid
			if x_dim and y_dim:
				# Get dataset to determine actual dimensions
				datasets = hdf.datasets()
				if datasets:
					first_ds_name = list(datasets.keys())[0]
					ds = hdf.select(first_ds_name)
					dims = ds.dimensions()
					if dims:
						# Assume first two dimensions are Y, X
						dim_names = list(dims.keys())
						if len(dim_names) >= 2:
							y_dim = dims[dim_names[0]]
							x_dim = dims[dim_names[1]]
					ds.endaccess()
			
			# Calculate pixel size based on tile dimensions
			# Standard MODIS tile is 1200x1200 km at various resolutions
			if '09Q1' in os.path.basename(hdf_file) or '13Q1' in os.path.basename(hdf_file):
				# 250m resolution
				pixel_size_x = 231.65635826395834
				pixel_size_y = 231.65635826395834
			elif '09A1' in os.path.basename(hdf_file):
				# 500m resolution  
				pixel_size_x = 463.31271652779167
				pixel_size_y = 463.31271652779167
			else:
				# Default 1km resolution
				pixel_size_x = 926.625433055833
				pixel_size_y = 926.625433055833
		
		# Build geotransform
		if ulc_lon is not None and ulc_lat is not None and pixel_size_x and pixel_size_y:
			geoTransform = (ulc_lon, pixel_size_x, 0.0, ulc_lat, 0.0, -pixel_size_y)
		else:
			# Fallback to default values if parsing failed
			geoTransform = None
	else:
		geoTransform = None
	
	# Get dimensions from a dataset
	datasets = hdf.datasets()
	if datasets:
		first_ds_name = list(datasets.keys())[0]
		ds = hdf.select(first_ds_name)
		dims = ds.dimensions()
		dim_values = list(dims.values())
		if len(dim_values) >= 2:
			rasterYSize = dim_values[0]
			rasterXSize = dim_values[1]
		else:
			rasterYSize, rasterXSize = None, None
		ds.endaccess()
	else:
		rasterYSize, rasterXSize = None, None
	
	hdf.end()
	
	return geoTransform, sr, rasterXSize, rasterYSize


def toRaster(in_array,out_path,model_file,dtype = None,*args,**kwargs) -> None:
	"""
	This function saves a numpy array into a raster file, with
	the same project and extents as the provided model file.

	As implemented, this function works ONLY for arrays that can
	be coerced to Int16 type.

	...

	Parameters
	----------

	in_array: numpy.array
		The array to be written to disk
	out_path: str
		Full path to raster file where the output will be written
	model_file: str
		Existing raster file with matching spatial reference and geotransform
	qa_array (optional): numpy.array
		If this parameter is used, the output raster will have two bands. Band
		1 stores in_array, band 2 stores qa_array
	"""

	# determine number of output bands
	if kwargs.get("qa_array") is not None:
		nbands = 2
	else:
		nbands = 1

	## extract extent, geotransform, and projection
	ext = os.path.splitext(model_file)[1]
	
	if ext == ".hdf":
		# Use pyhdf for HDF4 files
		try:
			geoTransform, sr, rasterXSize, rasterYSize = _get_hdf4_geotransform(model_file)
			
			# If pyhdf parsing failed, fall back to GDAL
			if geoTransform is None:
				log.warning("Failed to parse geotransform from HDF4 with pyhdf, falling back to GDAL")
				refDs = gdal.Open(model_file, 0)
				if refDs is None:
					raise octvi.exceptions.FileTypeError(f"Failed to open HDF4 file: {model_file}")
				sr = refDs.GetProjection()
				geoTransform = refDs.GetGeoTransform()
				
				# Handle GDAL subdataset structure
				if geoTransform[1] == 1.0:
					ds_sub = gdal.Open(refDs.GetSubDatasets()[0][0])
					geoTransform = ds_sub.GetGeoTransform()
					sr = ds_sub.GetProjection()
					ds_sub = None
				refDs = None
		except Exception as e:
			log.error(f"Error reading HDF4 file: {e}")
			raise
			
	elif ext == ".h5":
		# Use h5py for HDF5 files (VIIRS)
		pixelSize = 463.3127165
		try:
			with h5py.File(model_file, mode='r') as refDs:
				fileMetadata = refDs['HDFEOS INFORMATION']['StructMetadata.0'][()].split()
				fileMetadata = [m.decode('utf-8') for m in fileMetadata]
				ulc = [i for i in fileMetadata if 'UpperLeftPointMtrs' in i][0]
				ulcLon = float(ulc.split('=(')[-1].replace(')', '').split(',')[0])
				ulcLat = float(ulc.split('=(')[-1].replace(')', '').split(',')[1])

				# Special behavior for VNP09CMG
				if 'CMG' in os.path.basename(model_file):
					ulcLon = ulcLon / 1000000
					ulcLat = ulcLat / 1000000
					pixelSize = 0.05
				# VNP09A1 / VJ109A1 are 1km products
				elif '09A1' in os.path.basename(model_file):
					pixelSize = 926.625433055833

				geoTransform = (ulcLon, pixelSize, 0.0, ulcLat, 0.0, -pixelSize)
		except Exception as e:
			log.error(f"Error reading HDF5 metadata: {e}")
			# Fallback to GDAL
			refDs = gdal.Open(model_file, 0)
			if refDs is None:
				raise octvi.exceptions.FileTypeError(f"Failed to open HDF5 file: {model_file}")
			geoTransform = refDs.GetGeoTransform()
			refDs = None
		
		# Viirs sinusoidal projection
		sr = 'PROJCS["unnamed",GEOGCS["Unknown datum based upon the custom spheroid",DATUM["Not specified (based on custom spheroid)",SPHEROID["Custom spheroid",6371007.181,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Sinusoidal"],PARAMETER["longitude_of_center",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["Meter",1]]'
	else:
		# For other formats, use GDAL
		refDs = gdal.Open(model_file, 0)
		if refDs is None:
			raise octvi.exceptions.FileTypeError(f"Failed to open file: {model_file}")
		sr = refDs.GetProjection()
		geoTransform = refDs.GetGeoTransform()
		refDs = None
	
	# Get dimensions from array
	rasterYSize, rasterXSize = in_array.shape

	## parse datatype
	typeTable = {"Byte":gdal.GDT_Byte,"Int16":gdal.GDT_Int16,"Int32":gdal.GDT_Int32,"Float32":gdal.GDT_Float32,"Float64":gdal.GDT_Float64}
	outType = typeTable.get(dtype,gdal.GDT_Int16)
	if( kwargs.get("qa_array") is not None) and (outType not in [gdal.GDT_Int16,gdal.GDT_Int32]):
		log.warning("When qa_array is set in octvi.array.toRaster, dtype must be one of 'Int16' or 'Int32. Results will be coerced to Int16.")
		outType = gdal.GDT_Int16

	## write to disk
	driver = gdal.GetDriverByName('GTiff')
	dataset = driver.Create(out_path,rasterXSize,rasterYSize,nbands,outType,['COMPRESS=DEFLATE'])
	dataset.GetRasterBand(1).WriteArray(in_array)
	dataset.GetRasterBand(1).SetNoDataValue(-3000)
	if kwargs.get("qa_array") is not None:
		dataset.GetRasterBand(2).WriteArray(kwargs.get("qa_array"))
	dataset.SetGeoTransform(geoTransform)
	dataset.FlushCache() # Write to disk
	del dataset

	## project
	ds = gdal.Open(out_path,1)
	if ds:
		res = ds.SetProjection(sr)
		if res != 0:
			log.error("--projection failed: {}".format(str(res)))
		ds = None
	else:
		log.error("--could not open with GDAL")

	return None
