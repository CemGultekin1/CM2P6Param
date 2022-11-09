


FIELD_NAMES = 'u v T'.split()
FORCING_NAMES = 'Su Sv ST'.split()
LSRP0_NAMES = [f+'0' for f in FORCING_NAMES]
LSRP0_RES_NAMES = [f+'0_res'for f in FORCING_NAMES]
LSRP1_NAMES = [f+'1' for f in FORCING_NAMES]
LSRP1_RES_NAMES = [f+'1_res'for f in FORCING_NAMES]
LATITUDE_NAMES = ['abs_lat','sign_lat']

def rename(ds):
    # print(ds)
    varnames = list(ds.data_vars)
    for var in varnames:
        if 'temp' in var:
            ds = ds.rename({var:'T'})
        elif var == 'usurf':
            ds = ds.rename({var:'u'})
        elif var == 'vsurf':
            ds = ds.rename({var:'v'})
    coord_names = list(ds.dims.keys())
    coord_renames = {'xu_ocean':'ulon','yu_ocean':'ulat','xt_ocean':'tlon','yt_ocean':'tlat'}
    # print(ds)
    for key,val in coord_renames.items():
        if key in coord_names:
            ds = ds.rename(**{key:val})
    if 'st_ocean' in coord_names:
        ds = ds.rename({'st_ocean': 'depth'})
    # print(ds)
    return ds
def get_var_mask_name(key):
    return f"{key}_mask"
