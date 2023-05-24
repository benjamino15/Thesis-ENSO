########################### Adding map to causal diagram ##################################

import cartopy
import cartopy.crs as ccrs



# Create Plate Carrée projection
proj = ccrs.Orthographic(central_longitude=-155.0,central_latitude=15.0)
central_proj = ccrs.PlateCarree(central_longitude=0.0)

# Create a figure and axis using the Plate Carrée projection
fig, ax = plt.subplots(subplot_kw={'projection': proj})

# Add coastlines to the axis
ax.coastlines()
ax.add_feature(cartopy.feature.LAND, zorder=-1, facecolor = 'lightgrey')
ax.add_feature(cartopy.feature.OCEAN, zorder=-1, facecolor = 'white')
ax.add_feature(cartopy.feature.COASTLINE, zorder=-1, linewidth=.5)
#Add areas
x, y = [130, 150, 150, 130], [-5, -5, 5, 5]
ax.fill(x, y, transform=central_proj, color='coral', alpha=0.4, zorder=0)
x, y = [-150, -140, -130, -120, -120, -130, -140, -150], [-5, -5, -5, -5, 5, 5, 5, 5]
ax.fill(x, y, transform=central_proj, color='coral', alpha=0.4,zorder=0)
x, y = [-100, -80, -80, -100], [-5, -5, 5, 5]
ax.fill(x, y, transform=central_proj, color='coral', alpha=0.4, zorder=0)
ax.set_global()
ax.gridlines(linewidth=.2, zorder=0)


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