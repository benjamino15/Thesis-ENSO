import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy

# Create a NearsidePerspective projection centered on Chicago
# Create an AzimuthalEquidistant projection centered on the North Pole
proj = ccrs.Mollweide(central_longitude=-155.)


# Create a figure and axes with the projection
fig, ax = plt.subplots(subplot_kw={'projection': proj})

# Add map features
ax.coastlines()
ax.gridlines()


# Show the plot
plt.show()


rotated_crs = ccrs.RotatedPole(pole_longitude=-155.0, pole_latitude=-45.0)

ax = plt.axes(projection=rotated_crs)
#ax.set_extent([-40, 100, -20, 20], crs=ccrs.PlateCarree())
ax.coastlines(resolution='50m')
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

plt.show()

