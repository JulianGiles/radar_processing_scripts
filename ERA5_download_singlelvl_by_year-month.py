import cdsapi
import sys
from optparse import OptionParser
parser = OptionParser(usage="usage: %prog [year] [variable] ",\
                      version='%prog v1.0')
# general options
parser.add_option("-q", "--quiet",\
                   action="store_false", dest="verbose", default=True,\
                  help="don't print status messages to stdout")
(options, args) = parser.parse_args()
c = cdsapi.Client()
try:
    iyear = str(args[0])
except IndexError:
    sys.stderr.write("____ERROR____: Year not defined!!")
    sys.exit(200)
try:
    imonth = str(args[1])
except IndexError:
    sys.stderr.write("____ERROR____: Month not defined!!")
    sys.exit(200)
try:
    ivar=str(args[2])
except IndexError:
    sys.stderr.write("____ERROR____: Variable not defined!!")
    sys.exit(200)
try:
    north=float(args[3])
except IndexError:
    sys.stderr.write("____ERROR____: North not defined!!")
    sys.exit(200)
try:
    south=float(args[4])
except IndexError:
    sys.stderr.write("____ERROR____: South not defined!!")
    sys.exit(200)
try:
    west=float(args[5])
except IndexError:
    sys.stderr.write("____ERROR____: West not defined!!")
    sys.exit(200)
try:
    east=float(args[6])
except IndexError:
    sys.stderr.write("____ERROR____: East not defined!!")
    sys.exit(200)


if imonth == "all":
    months = [
            '01','02','03','04','05','06','07','08','09','10','11','12'
            ]
else:
    months=[imonth]

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type':'reanalysis',
        'data_format':'netcdf',
        'variable':ivar,
        'year':[
            iyear,
        ],
        'month':months,
        'day':[
            '01','02','03',
            '04','05','06',
            '07','08','09',
            '10','11','12',
            '13','14','15',
            '16','17','18',
            '19','20','21',
            '22','23','24',
            '25','26','27',
            '28','29','30',
            '31'
        ],
        'time':[
            '00:00','01:00','02:00',
            '03:00','04:00','05:00',
            '06:00','07:00','08:00',
            '09:00','10:00','11:00',
            '12:00','13:00','14:00',
            '15:00','16:00','17:00',
            '18:00','19:00','20:00',
            '21:00','22:00','23:00'
        ],
        'area':[
            north, west, south, east,
        ],
    },
    f'{ivar}_{iyear}-{imonth}.nc')
