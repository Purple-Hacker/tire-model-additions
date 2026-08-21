AIR_DENSITY = 1.184
G = 9.81
TIRE_RADIUS = 0.2032

# ttc data is measured on calspan's 3m safety-walk belt, peak mu ~2.3-2.7.
# real asphalt is roughly 2/3 of that. all tire model outputs get derated by
# this before the sim sees them. retune against skidpad data when we have it.
MU_SCALE = 0.65
