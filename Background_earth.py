########################### Adding map to causal diagram ##################################

import cartopy
import cartopy.crs as ccrs
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
red_patch = mpatches.Patch(color='coral', alpha = 0.4, label='Variables region')
blue_patch = mpatches.Patch(color='lightblue',alpha = 0.4, label='Niño 3.4 region')


# Create Plate Carrée projection
proj = ccrs.PlateCarree(central_longitude=180.0)
central_proj = ccrs.PlateCarree(central_longitude=0.0)

# Create a figure and axis using the Plate Carrée projection
fig, ax = plt.subplots(subplot_kw={'projection': proj})

# Add coastlines to the axis
ax.coastlines()
ax.add_feature(cartopy.feature.LAND, zorder=-1, facecolor = 'lightgrey')
ax.add_feature(cartopy.feature.OCEAN, zorder=-1, facecolor = 'white')
ax.add_feature(cartopy.feature.COASTLINE, zorder=-1, linewidth=.5)
#Add areas
x, y = [-170, -120, -120, -170], [-5.5, -5.5, 5.5, 5.5]
ax.fill(x, y, transform=central_proj, color='lightblue', alpha=0.4, zorder=0)
x, y = [130, 150, 150, 130], [-5, -5, 5, 5]
ax.fill(x, y, transform=central_proj, color='coral', alpha=0.4, zorder=0)
x, y = [-150, -140, -130, -120, -120, -130, -140, -150], [-5, -5, -5, -5, 5, 5, 5, 5]
ax.fill(x, y, transform=central_proj, color='coral', alpha=0.4,zorder=0)
x, y = [-100, -80, -80, -100], [-5, -5, 5, 5]
ax.fill(x, y, transform=central_proj, color='coral', alpha=0.4, zorder=0)
ax.set_global()
ax.gridlines(linewidth=.2, zorder=0)

ax.set_extent([80, -120, -40, 40], crs=ccrs.PlateCarree(central_longitude=-140.0))
plt.legend(handles=[red_patch, blue_patch])
ax.set_yticks([-40, -20, 0, 20, 40], crs=ccrs.PlateCarree())

plt.show()

# causal network plot

node_pos = {
    'y': np.array([0., 0., 0.]),
    'x': np.array([-5000000., 1000000., 6000000.])
}

ax = tp.plot_graph(
    fig_ax=(fig, ax),
    graph = results['graph'],
    node_pos=node_pos,
    figsize=(10, 5),
    val_matrix=results['val_matrix'],
    #cmap_edges='RdBu_r',
    #edge_ticks=.5,
    #show_colorbar=False,
    var_names=var_names,
    arrow_linewidth= 4,
    curved_radius=.35,
    node_size=1000000.,
    #node_label_size=8,
    node_aspect= 1.,
    #link_label_fontsize=0,
    #label_fontsize=8,
    node_colorbar_label='auto-MCI',
    link_colorbar_label='cross-MCI')

plt.show()