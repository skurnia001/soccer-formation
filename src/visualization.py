import numpy as np
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import row, widgetbox,column
from bokeh.models import ColumnDataSource, LabelSet, CustomJS, Title
from bokeh.models.widgets import Slider, Paragraph, Button, CheckboxButtonGroup
from bokeh.plotting import figure
from bokeh.io import show, output_notebook


def f(doc):
    source_coor_A = ColumnDataSource(
        data=dict(x=source_df_A['x_pos'], y=source_df_A['y_pos'], player_id=source_df_A['player_id']))
    source_coor_B = ColumnDataSource(
        data=dict(x=source_df_B['x_pos'], y=source_df_B['y_pos'], player_id=source_df_B['player_id']))

    plot = figure(name='base', plot_height=550, plot_width=850,
                  title="Game Animation", tools="reset,save,wheel_zoom,pan",
                  x_range=(0, 110), y_range=(0, 72), toolbar_location="below")

    plot.image_url(url=['background.png'], x=0, y=0, w=110, h=72, anchor='bottom_left')
    plot.scatter('x', 'y', source=source_coor_A, size=20)

    labels = LabelSet(x='x', y='y', text='player_id',
                      source=source_coor_A, y_offset=-8,
                      render_mode='canvas', text_color='black',
                      text_font_size="8pt", text_align='center')

    # https://github.com/samirak93/Game-Animation/blob/master/Animation/game_animation.py#L214
    """
    Remove plot background and alter other styles

    """

    def plot_clean(plot):
        plot.xgrid.grid_line_color = None
        plot.ygrid.grid_line_color = None
        plot.axis.major_label_text_font_size = "10pt"
        plot.axis.major_label_standoff = 0
        plot.border_fill_color = "white"
        plot.title.text_font = "times"
        plot.title.text_font_size = '10pt'
        plot.background_fill_color = "white"
        plot.title.align = 'center'
        return plot

    plot.add_layout(labels)
    plot.axis.visible = False
    plot = plot_clean(plot)

    slider_start = data_df.frame_id.min()
    slider_end = data_df.frame_id.max()
    game_time = Slider(title="Frame ID", value=slider_start, start=slider_start, end=slider_end, step=1)

    def update_data(attrname, old, new):
        slider_value = np.int32(game_time.value)

        source_df_A = data_A[data_A['frame_id'] == slider_value]
        source_df_B = data_B[data_B['frame_id'] == slider_value]

        source_coor_A.data = dict(x=source_df_A['x_pos'], y=source_df_A['y_pos'], player_id=source_df_A['player_id'])
        source_coor_B.data = dict(x=source_df_B['x_pos'], y=source_df_B['y_pos'], player_id=source_df_B['player_id'])

    for w in [game_time]:
        w.on_change('value', update_data)

    layout = column(plot, game_time)

    doc.add_root(layout)