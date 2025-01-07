import json

# Create a 2D dictionary
data = {
    "elev": {
        "staout_fname": "staout_1",
        "name": "zeta",
        "long_name": "water surface elevation above navd88",
        "stardard_name": "sea_surface_height_above_navd88",
        "units": "m",
    },
    "salinity": {
        "staout_fname": "staout_6",
        "name": "salinity",
        "long_name": "salinity",
        "stardard_name": "sea_water_salinity",
        "units": "psu",
    },
    "temperature": {
        "staout_fname": "staout_5",
        "name": "temperature",
        "long_name": "temperature",
        "stardard_name": "sea_water_temperature",
        "units": "c",
    },
    "u": {
        "staout_fname": "staout_7",
        "name": "u",
        "long_name": "Eastward Water Velocity",
        "stardard_name": "eastward_sea_water_velocity",
        "units": "meters s-1",
    },
    "v": {
        "staout_fname": "staout_8",
        "name": "v",
        "long_name": "Westward Water Velocity",
        "stardard_name": "westward_sea_water_velocity",
        "units": "meters s-1",
    },

}

# Convert the dictionary to JSON
json_data = json.dumps(data, indent=4)

with open("stofs_3d_atl_staout_nc.json", "w") as f:
    f.write(json_data)
